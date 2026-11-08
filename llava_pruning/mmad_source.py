"""Scoped MMAD inference identity and verified recovery of legacy whole-tree hashes.

An unrelated root experiment is not an inference dependency. Legacy manifests
store only an aggregate hash: resolve it to an EXACT Git snapshot before
comparing scoped dependencies. Never relabel an unknown historical hash.
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


def verify_source(spec,root):
    root=Path(root);files=working_files(root);current=scope_identity(files)
    saved=spec.get('mmad_source_identity')
    if saved is not None:
        if saved!=current:
            changed=', '.join(changed_paths(saved,current)[:8])
            raise ValueError('MMAD inference dependencies changed: '+changed+'; use the original inference code')
        return dict(identity=current)
    legacy=spec['source_identity']
    if legacy in (whole_identity(files),whole_identity(files,windows=True)):
        return dict(identity=current,legacy_recovery=dict(reference='unchanged working source',legacy_identity=legacy))
    cache={};visited=set()
    for revision in git(root,'rev-list','--max-count=100','HEAD').decode().splitlines():
        original=revision_files(root,revision,cache)
        signature=whole_identity(original)
        if signature['sha256'] in visited: continue
        visited.add(signature['sha256'])
        if legacy not in (signature,whole_identity(original,windows=True)): continue
        previous=scope_identity(original)
        if previous!=current:
            changed=', '.join(changed_paths(previous,current)[:8])
            raise ValueError(f'Historical source matches Git {revision[:12]}, but MMAD inference dependencies changed: {changed}')
        return dict(identity=current,legacy_recovery=dict(reference=revision,legacy_identity=legacy,
            verification='Exact legacy whole-source hash matched Git; scoped inference code is unchanged'))
    raise ValueError('Cannot identify the saved legacy source hash in the latest 100 Git revisions; '
                     'existing results are preserved. Restore the original source before resuming; no hash check was bypassed')
