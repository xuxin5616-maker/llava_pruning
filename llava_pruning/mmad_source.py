"""Scoped MMAD inference identity and verified recovery of legacy whole-tree hashes.

An unrelated root experiment is not an inference dependency. Legacy manifests
store only an aggregate hash: reconstruct it EXACTLY from Git, optionally
including unchanged local additions/edits, before comparing dependencies.
Never relabel an unknown historical hash or modify local source files.
"""

import ast
import hashlib
from pathlib import Path
import subprocess


POLICY="mmad-inference-imports-v1"
ENTRY="benchmark_mmad_binary.py"
FUNCTIONS={"schedule","make_trial_method","summarize_records","restore_records",
           "evaluate_samples","worker","metric_row"}
CONSTANTS={"FIELDS","SOURCE_FIELDS","HYBRID_COUNT_FIELDS","BUDGET_FIELDS","EXCLUDED_SOURCES"}
# Reviewed transition 67136cf -> 6aea384: the existing joint-self tail0
# becomes configurable, and the extension manifest carries that new option.
# These exact AST/file pairs, NOT filenames alone, define the compatibility.
TAIL0_FUNCTION_PAIRS={
    'schedule':('c42993536ae352d3cd70526b12454529c99401d2d29faafb19127bd0fe525f5a',
                '85f01d65ef12434b18a0540a618668f187c3ac78f4e052a91470a9fc510ca675'),
    'make_trial_method':('c0d19194e331f48144f483780b3c450474aba1f3e38159f61ca63eac111a5a50',
                         '0453547d9ad5ab0b6477b18679483cd639cc2122e50d5ecd72357b7033065756')}
EXTENSION='llava_pruning/mmad_extension.py'
TAIL0_EXTENSION_PAIR=('fe6f8f3b0c6cdd9af7d91227f62328bfb75b2b1c39e306eff9321b77e4f2628f',
                      '69d0fce97b11e5356b212f052acb8e5f1262f57b3c345347612fa56b584b4b2f')
# Reviewed file histories. MMAD's default attachment and depth are preserved.
STUDENT_LAYER_HASHES = {
    'llava_pruning/attention_predictor.py': (
        '64a49902aeb3e374ab77406da12918339ba960df011b6a061fd0d45eceea6b5e',
        '898f54ef83cb68243a3850e96d3d3d2276798d5c39f255bca2c950f3c8693f99',
        '7be508bfd41bb105dcf85ec73d2ddbc255cde0600b909335766284a3ad4788ed'),
    'llava_pruning/pool_distill.py': (
        'cd0f64d81bcd4e0f07fc1563741900bbcd7b70d928c68b54da4e8314f5d68c19',
        'f21349cee7bcc4275310850ba19ed7f8b2cb29cb4fbee66c594e3da32583b5dc'),
    'llava_pruning/pool_lean.py': (
        '419dda1e61e2981e718699da52a0ad8c946d86e5e2c8c34d8703e27472dc06e9',
        '114136631883422d28a826bc0b58e7471146cdcad5d06890df44516398358452'),
    'llava_pruning/visa_comparison_methods.py': (
        '8c57806b4541d70791f5d41b87be18a8c7ebaae84b62cce7db9637b6204ce8ab',
        'e47da65c53ed2bedbc490fc48a3299ec896caaa839b08ce1614c9a7b3a8f4ed8'),
    'vendor/llava/llava/model/language_model/attention_predictor.py': (
        '5a16c112a87c23125b182e4bc738b11f5294ac9d2c9f895e5c089941bf6c8022',
        '82abf3cc49b89b14528c013c19e6f317ffbe2f611aadd0495973d6ac26343e0e',
        '1a5b941fda5b5c58a192a768cf51f79544544ef3e66ff07b1b26f12ee2616342'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def eligible(path):
    return path.endswith('.py') and ('/' not in path or
        path.startswith(('llava_pruning/','vendor/llava/llava/','third_party/')))


def working_files(root):
    files=list(root.glob('*.py'))
    for name in ('llava_pruning','vendor/llava/llava','third_party'):
        files.extend((root/name).rglob('*.py'))
    return {p.relative_to(root).as_posix():p.read_bytes().replace(b'\r\n',b'\n') for p in files}


def whole_identity(files, *, windows=False):
    paths=sorted(files,key=str.casefold if windows else None)
    return dict(files=len(paths),sha256=digest('\n'.join(p+':'+digest(files[p]) for p in paths).encode()))


def entry_tree(data):
    tree=ast.parse(data)
    nodes=[]
    for node in tree.body:
        if isinstance(node,(ast.Import,ast.ImportFrom)) or (isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in FUNCTIONS):
            nodes.append(node)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in CONSTANTS for t in node.targets):
            nodes.append(node)
    if {n.name for n in nodes if isinstance(n,ast.FunctionDef)}!=FUNCTIONS:
        raise ValueError('Historical MMAD inference entry is unsupported')
    return ast.Module(body=nodes,type_ignores=[])


def scope_identity(files):
    """Follow static local imports, including imports INSIDE functions/classes.

    All native/vendor and third-party Python files are conservative roots for
    runtime model dispatch. Entry parsing/setup/source-validation code is not
    model inference; actual scheduling, generation and metric functions are.
    """
    pending=[ENTRY]+[p for p in files if p.startswith(('vendor/llava/llava/','third_party/'))]
    seen={}
    def add_module(module):
        parts=module.split('.')
        for size in range(1,len(parts)):
            init='/'.join(parts[:size])+'/__init__.py'
            if init in files and init not in seen: pending.append(init)
        for path in (module.replace('.','/')+'.py',module.replace('.','/')+'/__init__.py'):
            if path in files and path not in seen: pending.append(path)
    while pending:
        path=pending.pop()
        if path in seen: continue
        if path not in files: raise ValueError(f'Missing MMAD inference dependency: {path}')
        data=files[path]
        tree=entry_tree(data) if path==ENTRY else ast.parse(data)
        seen[path]=digest(ast.dump(tree,include_attributes=False).encode()) if path==ENTRY else digest(data)
        package=path[:-3].replace('/','.').split('.')
        if package[-1]=='__init__': package=package[:-1]
        else: package=package[:-1]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                for alias in node.names: add_module(alias.name)
            elif isinstance(node,ast.ImportFrom):
                prefix=package[:len(package)-node.level+1] if node.level else []
                module='.'.join([*prefix,*([node.module] if node.module else [])])
                if module: add_module(module)
                for alias in node.names:
                    if alias.name!='*': add_module(module+'.'+alias.name if module else alias.name)
    entries=dict(sorted(seen.items()))
    return dict(policy=POLICY,files=entries,sha256=digest('\n'.join(k+':'+v for k,v in entries.items()).encode()))


def git(root,*args,input=None):
    result=subprocess.run(['git','-C',str(root),*args],input=input,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if result.returncode: raise ValueError('Cannot verify historical source using Git: '+result.stderr.decode(errors='replace').strip())
    return result.stdout


def revision_files(root,revision,cache):
    paths={}
    for row in git(root,'ls-tree','-r','-z',revision).split(b'\0'):
        if not row: continue
        header,path=row.split(b'\t',1);path=path.decode('utf-8')
        if eligible(path): paths[path]=header.split()[2].decode('ascii')
    missing=list(dict.fromkeys(oid for oid in paths.values() if oid not in cache))
    if missing:
        raw=git(root,'cat-file','--batch',input=('\n'.join(missing)+'\n').encode());offset=0
        for oid in missing:
            end=raw.index(b'\n',offset);header=raw[offset:end].split()
            if len(header)!=3 or header[0].decode()!=oid or header[1]!=b'blob':
                raise ValueError('Invalid Git source blob')
            size=int(header[2]);offset=end+1
            cache[oid]=raw[offset:offset+size].replace(b'\r\n',b'\n');offset+=size+1
    return {path:cache[oid] for path,oid in paths.items()}


def changed_paths(before,after):
    left,right=before['files'],after['files']
    return [p for p in sorted(set(left)|set(right)) if left.get(p)!=right.get(p)]


def default_student_layer_updates(before, after):
    return [path for path, hashes in STUDENT_LAYER_HASHES.items()
            if before['files'].get(path) in hashes[:-1] and after['files'].get(path) == hashes[-1]]


def local_changes(files,head):
    """Use file contents, including ignored Python files seen by the old guard.

    JSON/shell/backup files were never in the old Python-only identity. Files
    introduced by later commits are NOT local additions just because an older
    revision lacks them: compare working files with the CURRENT Git HEAD.
    """
    additions={p:data for p,data in files.items() if p not in head}
    modified={p:data for p,data in files.items() if p in head and data!=head[p]}
    deleted=set(head)-set(files)
    return additions,modified,deleted


def legacy_candidates(original,additions,modified,deleted):
    yield original,dict(mode='git_only',added=[],modified=[],deleted=[])
    if additions:
        yield dict(original,**additions),dict(mode='git_plus_local_additions',
            added=sorted(additions),modified=[],deleted=[])
    if modified or deleted:
        combined=dict(original,**additions,**modified)
        for path in deleted: combined.pop(path,None)
        yield combined,dict(mode='git_plus_local_changes',added=sorted(additions),
            modified=sorted(modified),deleted=sorted(deleted))


def tail0_compatibility(previous,files,spec,current):
    """Prove the reviewed default-tail0 extension is inert for THIS saved run.

    Resume restores the frozen job list; it does not call schedule again.
    The method's old literal tail0 and new job.get(...,0) are equivalent only
    when every saved joint-self job is tail0. All other dependency hashes,
    generation/metrics functions and method code must remain unchanged.
    No old source is executed or written to disk.
    """
    jobs=spec.get('jobs');tails=spec.get('joint_self_tails',[0])
    if not isinstance(jobs,list) or not jobs or not isinstance(tails,(list,tuple)) or list(tails)!=[0]: return None
    for job in jobs:
        if job.get('variant')=='joint_self':
            tail=job.get('tail_drop_layers',0)
            if type(tail) is not int or tail!=0: return None
    earlier=scope_identity(previous)
    if set(changed_paths(earlier,current))-{ENTRY,EXTENSION}: return None
    old_tree,new_tree=entry_tree(previous[ENTRY]),entry_tree(files[ENTRY])
    before={n.name:n for n in old_tree.body if isinstance(n,ast.FunctionDef)}
    after={n.name:n for n in new_tree.body if isinstance(n,ast.FunctionDef)}
    for name,pair in TAIL0_FUNCTION_PAIRS.items():
        actual=tuple(digest(ast.dump(nodes[name],include_attributes=False).encode()) for nodes in (before,after))
        if actual!=pair: return None
    if (digest(previous.get(EXTENSION,b'')),digest(files.get(EXTENSION,b'')))!=TAIL0_EXTENSION_PAIR: return None
    # Substitute ONLY the four exactly reviewed changes, then demand equality
    # of the complete inference identity. Any unrelated scoring/forward/
    # timing/statistics change still fails the normal dependency check.
    old_tree.body=[after[n.name] if isinstance(n,ast.FunctionDef) and n.name in TAIL0_FUNCTION_PAIRS else n
                   for n in old_tree.body]
    adjusted=dict(previous)
    adjusted[ENTRY]=ast.unparse(old_tree).encode('utf-8')
    adjusted[EXTENSION]=files[EXTENSION]
    if scope_identity(adjusted)!=current: return None
    return dict(transition='67136cf_to_6aea384_default_tail0',
        reviewed_files=[ENTRY,EXTENSION],joint_self_tails=[0],
        verification='Exact reviewed AST/file pairs; saved jobs unchanged; tail0 constructor equivalent; all other inference dependencies identical')


def verify_source(spec,root):
    root=Path(root);files=working_files(root);current=scope_identity(files)
    saved=spec.get('mmad_source_identity')
    if saved is not None:
        if saved==current:
            return dict(identity=current)
        updated=default_student_layer_updates(saved,current)
        remaining=[p for p in changed_paths(saved,current) if p not in updated]
        if remaining or not updated or saved.get('policy')!=current['policy']:
            changed=', '.join((remaining or changed_paths(saved,current))[:8])
            raise ValueError('MMAD inference dependencies changed: '+changed+'; use the original inference code')
        return dict(identity=current, compatibility=dict(transition='default_student_layer_2',reviewed_files=updated))
    legacy=spec['source_identity']
    if legacy in (whole_identity(files),whole_identity(files,windows=True)):
        return dict(identity=current,legacy_recovery=dict(reference='unchanged working source',legacy_identity=legacy))
    cache={};visited=set()
    head=revision_files(root,'HEAD',cache)
    additions,modified,deleted=local_changes(files,head)
    for revision in git(root,'rev-list','--max-count=100','HEAD').decode().splitlines():
        original=revision_files(root,revision,cache)
        for candidate,overlay in legacy_candidates(original,additions,modified,deleted):
            if len(candidate)!=legacy.get('files'): continue
            signature=whole_identity(candidate)
            if signature['sha256'] in visited: continue
            visited.add(signature['sha256'])
            if legacy not in (signature,whole_identity(candidate,windows=True)): continue
            previous=scope_identity(candidate)
            compatibility=None
            if previous!=current:
                updated=default_student_layer_updates(previous,current)
                adjusted=dict(candidate,**{p:files[p] for p in updated})
                adjusted_scope=scope_identity(adjusted) if updated else previous
                if adjusted_scope!=current:
                    compatibility=tail0_compatibility(adjusted,files,spec,current)
                if adjusted_scope!=current and compatibility is None:
                    changed=', '.join(changed_paths(adjusted_scope,current)[:8])
                    raise ValueError(f'Historical source matches Git {revision[:12]} ({overlay["mode"]}), '
                                     f'but MMAD inference dependencies changed: {changed}')
                if updated:
                    if compatibility is None:
                        compatibility=dict(transition='default_student_layer_2',reviewed_files=updated)
                    else:
                        compatibility['default_student_layer_updates']=updated
            return dict(identity=current,legacy_recovery=dict(reference=revision,legacy_identity=legacy,
                local_overlay=overlay,compatibility=compatibility,
                verification='Exact legacy whole-source hash matched reconstructed Git/local files; inference is unchanged or matches an exact reviewed default transition'))
    raise ValueError('Cannot identify the saved legacy source hash in the latest 100 Git revisions plus current local Python files; '
        f'saved_files={legacy.get("files")}, current_files={len(files)}, local_additions={len(additions)}, '
        f'local_modified={len(modified)}, local_deleted={len(deleted)}. '
        'Existing results are preserved; no hash check was bypassed')
