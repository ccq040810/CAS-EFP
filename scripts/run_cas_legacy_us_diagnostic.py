"""DIAGNOSTIC ONLY: replay the legacy integer lookup with microsecond indexes.

This explicitly preserves the old future-history failure. Hyperparameters,
targets, split boundaries and missing-value handling remain unchanged.
"""
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_cas_legacy_diagnostic as base

PROTOCOL='legacy_integer_lookup_FORCED_US_DIAGNOSTIC_ONLY_v1'
DEFAULT_ROOT=ROOT/'results/legacy_us_protocol_diagnostic_20261028'
NATIVE_ROOT=ROOT/'results/legacy_protocol_diagnostic_20261028'
NativeIndex=base.LegacyIndex


class MicrosecondLegacyIndex(NativeIndex):
    def __init__(self,index):
        super().__init__(index.as_unit('us'))


def reusable_audit(audit):
    origins=audit.get('origins',[])
    return bool(origins) and all(
        r['index_dtype'].startswith('datetime64[us,') and r['future_history']
        and r['valid_history'] and r['legacy_position']>r['correct_position']
        for r in origins)


def mark_audit(path):
    if not path.exists():return
    audit=base.transfer.read(path)
    audit.update(requested_lookup_unit='us',diagnostic_protocol=PROTOCOL)
    for r in audit.get('origins',[]):
        r['original_index_dtype']=r['index_dtype']
        r['effective_lookup_unit']='us'
    base.transfer.write(path,audit)


def configure():
    base.PROTOCOL=PROTOCOL
    base.LegacyIndex=MicrosecondLegacyIndex
    original_prepare=base.prepare_child
    original_summarize=base.summarize
    def prepare(args):
        final=args.output_root/'prepared'/args.source/args.market/f'window_{args.window}'
        old=NATIVE_ROOT/'prepared'/args.source/args.market/f'window_{args.window}'
        reused=False
        # Reuse only verified, completed Sundial features already produced with
        # microsecond lookup; causal ns Moirai features must be recomputed.
        if args.source=='cas_sundial' and not final.exists() and (old/'HISTORY_AUDIT.json').exists():
            audit=base.transfer.read(old/'HISTORY_AUDIT.json')
            manifest=base.transfer.read(old/'manifest.json')
            if reusable_audit(audit) and base.transfer.sha(old/'features.npz')==manifest['arrays_sha256']:
                for path,digest in manifest['source_sha256'].items():
                    if base.transfer.sha(path)!=digest:raise ValueError(f'Reused source changed: {path}')
                staging=args.output_root/'staging'/uuid.uuid4().hex
                staging.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(old,staging)
                manifest.update(diagnostic_protocol=PROTOCOL,requested_lookup_unit='us',
                                reused_from=str(old),reused_arrays_sha256=manifest['arrays_sha256'])
                base.transfer.write(staging/'manifest.json',manifest)
                final.parent.mkdir(parents=True,exist_ok=True)
                staging.rename(final)
                reused=True
                print('REUSE verified us history',old,flush=True)
        try:
            if not reused:original_prepare(args)
        finally:
            mark_audit(args.output_root/'history_audits'/args.source/args.market/f'window_{args.window}.json')
            mark_audit(final/'HISTORY_AUDIT.json')
        manifest=base.transfer.read(final/'manifest.json')
        manifest.update(requested_lookup_unit='us',diagnostic_protocol=PROTOCOL)
        base.transfer.write(final/'manifest.json',manifest)
    def summarize(args,jobs,failures):
        original_summarize(args,jobs,failures)
        import pandas as pd
        for path in args.output_root.glob('DIAGNOSTIC_*.csv'):
            table=pd.read_csv(path)
            table['diagnostic_protocol']=PROTOCOL
            table['lookup_unit']='us'
            table.to_csv(path,index=False)
        path=args.output_root/'DIAGNOSTIC_RESULTS.md'
        text=path.read_text()
        text=text.replace('# 旧协议诊断结果','# 统一微秒旧故障回放诊断结果')
        text=text.replace('不同环境原始日期单位可能不同，逐起报点记录于HISTORY_AUDIT.json；不进行填充或时间修复。',
            '旧整数查找所用的时间索引统一为微秒；审计同时记录原始精度与有效查找精度。不填充历史，不修复旧整数查找。')
        path.write_text(text,encoding='utf-8')
        completion=base.transfer.read(args.output_root/'completion.json')
        completion.update(diagnostic_protocol=PROTOCOL,lookup_unit='us')
        base.transfer.write(args.output_root/'completion.json',completion)
    base.prepare_child=prepare
    base.summarize=summarize
    # Child commands must re-enter this wrapper, not the native environment run.
    base.__file__=str(Path(__file__).resolve())


def main():
    if not any(a.split('=',1)[0]=='--output-root' for a in sys.argv[1:]):
        sys.argv.extend(['--output-root',str(DEFAULT_ROOT)])
    configure()
    base.main()


if __name__=='__main__':main()
