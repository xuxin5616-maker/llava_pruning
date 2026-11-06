"""Official MMAD Anomaly Detection questions, single query image / 0-shot.

Keep question/option text, including Maybe/Unknown distractors. Option letters
are not binary labels. GT/reference/mask annotations never enter the prompt.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re

from .data import Sample


SOURCES={'DS-MVTec':'MVTec-AD','MVTec-AD':'MVTec-AD','MVTec-LOCO':'MVTec-LOCO','VisA':'VisA','GoodsAD':'GoodsAD'}
TASK='Anomaly Detection'
OPTION=re.compile(r'^\s*(?:answer\s*:\s*)?\(?([A-E])\)?(?:[\s.:,\-]|$)',re.IGNORECASE)


def relative_image(value):
    raw=str(value).replace('\\','/');path=PurePosixPath(raw)
    if not raw or path.is_absolute() or '..' in path.parts or ':' in raw or len(path.parts)<5:
        raise ValueError(f'Invalid MMAD relative image path: {value!r}')
    if path.parts[0] not in SOURCES: raise ValueError(f'Unknown MMAD source: {path.parts[0]}')
    return path.as_posix()


def find_annotations(root, explicit=None):
    paths=[Path(explicit).expanduser().resolve()] if explicit else [root/p for p in
        ('mmad.json','dataset/MMAD/mmad.json','MMAD/mmad.json','annotations/mmad.json')]
    found={p.resolve() for p in paths if p.is_file()}
    if len(found)!=1:
        raise ValueError('Need exactly one official mmad.json; specify --annotations /path/to/mmad.json. '
                         'metadata.csv and MMAD-BBox annotations cannot identify these questions reliably.')
    return found.pop()


def resolve_image(root, relative, image_root=None):
    roots=[Path(image_root).expanduser().resolve()] if image_root else [root,root/'images',root/'dataset/MMAD',root/'MMAD']
    found={(base/relative).resolve() for base in roots if (base/relative).is_file()}
    if len(found)!=1:
        raise ValueError(f'MMAD image has {len(found)} candidate files: {relative}; use --image-root to select the image root')
    # Official instructions explicitly permit source-directory symlinks.
    return found.pop()


def yes_no(value):
    value=str(value).strip().lower().rstrip('.!').strip()
    return value if value in {'yes','no'} else None


def prompt(record):
    return ('You are an industrial inspector who checks products by images.\n'
        +record['question']+'\n'+''.join(f'{k}. {v}\n' for k,v in record['options'].items())
        +"Answer with the option's letter from the given choices directly.")


def load_binary(data_root, annotations=None, image_root=None, seed=42, *, exclude_sources=()):
    root=Path(data_root).expanduser().resolve()
    if not root.is_dir(): raise NotADirectoryError(root)
    excluded=set(exclude_sources)
    if excluded-set(SOURCES.values()): raise ValueError('Unknown MMAD source exclusion')
    path=find_annotations(root,annotations)
    raw=path.read_bytes();data=json.loads(raw)
    if not isinstance(data,dict): raise ValueError('MMAD annotation must map relative image paths to question records')
    records=[];counts=Counter();skipped=Counter();excluded_counts=Counter();extra=0
    rng=random.Random(seed)
    for key,item in sorted(data.items()):
        relative=relative_image(key)
        source=SOURCES[PurePosixPath(relative).parts[0]]
        if not isinstance(item,dict): raise ValueError(f'Invalid MMAD record: {relative}')
        conversations=[(name,value) for name,value in item.items() if name.startswith('conversation')]
        if not conversations: skipped['no_conversation']+=1;continue
        # The official examples use the first conversation* field, not every
        # alternate annotation as another test image.
        name,questions=conversations[0]
        if not isinstance(questions,list): raise ValueError('MMAD conversation must be a list')
        for index,qa in enumerate(questions):
            if not isinstance(qa,dict): raise ValueError('Malformed MMAD question')
            if qa.get('type')!=TASK:
                skipped[str(qa.get('type','missing_type'))]+=1;continue
            if source in excluded:
                excluded_counts[source]+=1
                continue
            options=qa.get('Options');answer=qa.get('Answer');question=qa.get('Question')
            if (not isinstance(options,dict) or not 2<=len(options)<=5 or answer not in options or
                    not isinstance(question,str) or not question.strip() or
                    any(not isinstance(k,str) or not isinstance(v,str) for k,v in options.items())):
                raise ValueError(f'Malformed binary question: {relative}, {index}')
            binary=[yes_no(v) for v in options.values()]
            if binary.count('yes')!=1 or binary.count('no')!=1 or yes_no(options[answer]) is None:
                raise ValueError(f'Anomaly Detection requires Yes/No ground truth and one option each: {relative}')
            # Used for confusion groups only, as in official normal-vs-abnormal
            # scoring. Option-to-class mapping is evaluated AFTER generation.
            gt=0 if PurePosixPath(relative).parts[-2].lower() in {'good','normal'} else 1
            image=resolve_image(root,relative,image_root)
            shuffled=list(options.items());rng.shuffle(shuffled)
            mapping={chr(65+i):old_key for i,(old_key,_) in enumerate(shuffled)}
            choices={chr(65+i):text for i,(_,text) in enumerate(shuffled)}
            correct=next(k for k,v in mapping.items() if v==answer)
            record=dict(question_id=f'{relative}::{name}:{index}',image=relative,resolved_image=str(image),
                origin_path=relative,category=PurePosixPath(relative).parts[1],dataset=source,
                raw_source=PurePosixPath(relative).parts[0],question=question,options=choices,
                correct_option=correct,option_key_mapping=mapping,gt=gt,task=TASK)
            record['text']=prompt(record)
            records.append(record);counts[source]+=1;extra+=len(options)>2
    if not records: raise ValueError('No Anomaly Detection questions found in mmad.json')
    records.sort(key=lambda r:r['question_id'])
    return records,dict(annotations=str(path),annotations_sha256=hashlib.sha256(raw).hexdigest(),
        query_images=len(data),selected_query_images=len({r['image'] for r in records}),
        binary_questions=len(records),source_counts=dict(counts),excluded_sources=sorted(excluded),
        excluded_binary_questions=dict(excluded_counts),
        binary_questions_with_distractors=extra,skipped_question_types=dict(skipped),shots=0,
        option_policy='Question and option text retained; labels shuffled ONCE with seed, shared by all methods',
        conversation_policy='First conversation* field, matching the official example parser',
        gt_policy='normal=good/normal query directory, otherwise abnormal; only used for scoring',
        reference_images_used=False,masks_used=False)


def load_frozen(path, *, exclude_sources=()):
    rows=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    samples=[];seen=set()
    for row in rows:
        if row['question_id'] in seen: raise ValueError('Duplicate MMAD question ID')
        seen.add(row['question_id']);relative=relative_image(row['image'])
        source=SOURCES[PurePosixPath(relative).parts[0]]
        if row['dataset']!=source: raise ValueError('Frozen MMAD source differs from image path')
        if source in exclude_sources: raise ValueError(f'Excluded MMAD source in frozen input: {source}')
        image=Path(row['resolved_image'])
        if not image.is_file(): raise FileNotFoundError(image)
        if row['text']!=prompt(row): raise ValueError('Frozen question/options changed')
        samples.append(Sample(row['question_id'],image,row['category'],row['origin_path'],row['text'],
                              row['gt'],None,None,row))
    return samples


def parse_option(answer, options):
    match=OPTION.match(answer)
    if match and match.group(1).upper() in options: return match.group(1).upper()
    exact=[k for k,v in options.items() if answer.strip().lower().rstrip('.!')==v.strip().lower().rstrip('.!')]
    if len(exact)==1: return exact[0]
    return None


def canonical_answer(row, raw_answer):
    key=parse_option(raw_answer,row['options'])
    if key is None or yes_no(row['options'][key]) is None: return '',key
    prediction=row['gt'] if key==row['correct_option'] else 1-row['gt']
    return 'A' if prediction else 'B',key
