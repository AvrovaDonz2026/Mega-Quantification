#!/usr/bin/env python3
"""Sequential cold-cache MSE native-prefix sensitivity experiment, no raw output."""
from __future__ import annotations
import argparse
import json
import socket
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

IMAGE = 'sha256:e98ca3c6444896d8800dc63cad3d4d22ee842af32fb16badd6cc04ee9090babf'
NATIVE_SHA = '8e151d76fdd25603216aeb79c59a39d637ad25c47efd87a3581f6b65b3dc7218'
PROMPT_SHA = '20e75efb3ff7b137eee5484d35b30e0bb3668d8cc810b36990c7d3cae662d436'
VARIANTS = {'original': '4a667614d810f7b5570be23eb75956c603a85c3a', 'all-beta': '49d3acfd4d54b0cf1afffcfeb369c63af4f56499'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--remote-root', type=Path, required=True)
    p.add_argument('--native-private', type=Path, required=True)
    p.add_argument('--budget', type=int, default=512)
    o = p.parse_args()
    o.root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(o.remote_root / 'candidate-bundle-r1/parity-v7'))
    import run_v7 as v
    def get(port, path):
        with urllib.request.urlopen(f'http://127.0.0.1:{port}{path}', timeout=30) as r:
            return json.load(r)
    def progress(state, **fields):
        v.write_json(o.root / 'status.json', {'state': state, 'at_utc': v.now(), **fields})
    progress('preflight')
    try:
        if v.sha256_file(o.native_private) != NATIVE_SHA:
            raise RuntimeError('native-private SHA differs')
        native = json.loads(o.native_private.read_text())
        if len(native['cases']) != 1 or native['cases'][0]['case'] != 3:
            raise RuntimeError('anonymous case identity differs')
        old = native['cases'][0]
        prompt = old['request']['input_ids']; answer = old['response']['output_ids']
        if len(prompt) != 288 or v.digest(prompt) != PROMPT_SHA or len(answer) != 49152:
            raise RuntimeError('native prompt/output identity differs')
        if any(type(x) is not int or x < 0 for x in prompt + answer):
            raise RuntimeError('invalid native token IDs')
        if not all(answer[i] == answer[i-2032] for i in range(33131+2032, len(answer))):
            raise RuntimeError('historical native cycle no longer verifies')
        cut = 41472
        if not 1 <= o.budget <= 4096 or cut + o.budget > len(answer):
            raise RuntimeError('unsupported continuation budget')
        ids = prompt + answer[:cut]; reference = answer[cut:cut+o.budget]
        report = {'schema_version': 1, 'started_at_utc': v.now(), 'case': 'anonymous-case3',
                  'native_private_sha256': NATIVE_SHA, 'prompt_ids_sha256': PROMPT_SHA,
                  'input_ids_sha256': v.digest(ids), 'input_tokens': len(ids), 'native_prefix_tokens': cut,
                  'native_reference_sha256': v.digest(reference), 'historical_period_tokens': 2032,
                  'max_new_tokens': o.budget, 'repetitions': 2, 'image_id': IMAGE,
                  'scope': 'Fresh-prefill native-prefix sensitivity; does not reconstruct historical incremental SSM/KV state or establish natural generation repair.',
                  'protocol': {'context_length': 65536, 'temperature': 0, 'tp_size': 1, 'mtp': False,
                               'kv_cache_dtype': 'fp8_e4m3', 'mamba_ssm_dtype': 'float32',
                               'disable_radix_cache': True, 'allow_auto_truncate': False,
                               'cuda_graphs': False, 'FLA_USE_FAST_OPS': '0', 'force_shared_decode': '0'},
                  'arms': {}}
        model = o.remote_root / 'mse-model-view-r1/models/w4a4'
        model_identity = o.remote_root / 'mse-model-view-r1/model-identity.json'
        # Read termination IDs without decoding any question or response.
        cfg = json.loads((model / 'config.json').read_text())
        eos = cfg.get('text_config', cfg).get('eos_token_id', [])
        eos = [eos] if type(eos) is int else eos
        tok = json.loads((model / 'tokenizer.json').read_text())
        think_end = [x['id'] for x in tok.get('added_tokens', []) if x.get('content') == '</think>']
        for variant, revision in VARIANTS.items():
            arm_root = o.root / variant; arm_root.mkdir()
            a = SimpleNamespace(root=arm_root, job_id='mse-prefix-'+variant, model_scheme='w4a4',
                source_variant=variant, source_revision=revision, source=o.remote_root/f'source-package/sglang-{variant}',
                source_archive=o.remote_root/f'source-package/sglang-{variant}.tar.gz',
                source_manifest=o.remote_root/f'source-package/sglang-{variant}.manifest.json',
                model=model, model_provenance=model_identity, image=IMAGE, image_id=IMAGE,
                fla_use_fast_ops='0', force_shared_decode='0', health_timeout=1200)
            progress('preflight', arm=variant)
            source = v.source_gate(a); source.pop('source_file_hashes', None)
            model_proof = v.model_gate(a)
            port = 32252
            with socket.socket() as s:
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    raise RuntimeError('experiment port occupied')
            if v.run(['docker','image','inspect','--format','{{.Id}}', IMAGE]).stdout.strip() != IMAGE:
                raise RuntimeError('image identity differs')
            v.gpu_idle()
            command = v.server_command(a, a.job_id, 'baseline', port)
            command[command.index('--context-length')+1] = '65536'
            command += ['--disable-radix-cache']
            server = None
            arm = {'source_proof': source, 'model_proof': model_proof, 'trials': []}
            report['arms'][variant] = arm
            try:
                progress('starting', arm=variant)
                server = v.run(command).stdout.strip()
                v.owned(server, a.job_id, 'baseline', IMAGE, revision)
                info = v.wait_health(a, server, 'baseline', port)
                model_info = get(port, '/model_info')
                expected = {'host':'127.0.0.1', 'port':port, 'model_path':'/models/Qwen3.8-27B-NVFP4-W4A4',
                    'quantization':'modelopt_fp4', 'fp4_gemm_runner_backend':'flashinfer_cutlass',
                    'attention_backend':'flashinfer', 'sampling_backend':'pytorch', 'linear_attn_backend':'triton',
                    'mamba_backend':'triton', 'mamba_ssm_dtype':'float32', 'kv_cache_dtype':'fp8_e4m3',
                    'context_length':65536, 'tp_size':1, 'max_running_requests':1, 'enable_multimodal':True,
                    'speculative_algorithm':None, 'disable_radix_cache':True, 'allow_auto_truncate':False}
                if any(info.get(k) != value for k, value in expected.items()):
                    raise RuntimeError('server identity mismatch: '+','.join(k for k,val in expected.items() if info.get(k)!=val))
                if any(info.get(k) for k in ['dump_requests_folder','crash_dump_folder','log_requests']):
                    raise RuntimeError('server output logging enabled')
                if any(info.get('cuda_graph_config',{}).get(phase,{}).get('backend') != 'disabled' for phase in ['prefill','decode']):
                    raise RuntimeError('CUDA graph enabled')
                if model_info.get('model_path') != expected['model_path']:
                    raise RuntimeError('model path differs')
                arm['server_identity'] = {'passed':True, 'expected':expected, 'server_info_sha256':v.digest(info), 'model_info_sha256':v.digest(model_info)}
                runtime = v.runtime_source_hashes(a, server)
                v.write_json(arm_root / 'runtime-source.json', runtime)
                if not runtime['passed']: raise RuntimeError('runtime source hash differs')
                arm['runtime_source_sha256'] = v.sha256_file(arm_root / 'runtime-source.json')
                for trial in (1,2):
                    progress('request', arm=variant, trial=trial)
                    req = urllib.request.Request(f'http://127.0.0.1:{port}/flush_cache?timeout=180', data=b'{}', method='POST', headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req, timeout=200) as r:
                        if r.status != 200 or not r.read().startswith(b'Cache flushed.'):
                            raise RuntimeError('flush acknowledgement differs')
                    payload = {'input_ids':ids, 'sampling_params':{'temperature':0.0,'max_new_tokens':o.budget},
                               'stream':False, 'return_logprob':True, 'logprob_start_len':-1,
                               'return_text_in_logprobs':False, 'return_prompt_token_ids':True, 'log_metrics':False}
                    req = urllib.request.Request(f'http://127.0.0.1:{port}/generate', data=json.dumps(payload).encode(), method='POST', headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req, timeout=2400) as r: response = json.load(r)
                    out = response['output_ids']; meta = response['meta_info']
                    if not out or any(type(x) is not int or x < 0 for x in out) or len(out) != meta.get('completion_tokens') or len(out)>o.budget:
                        raise RuntimeError('invalid response token counts')
                    if response.get('prompt_token_ids') != ids or meta.get('prompt_tokens') != len(ids) or meta.get('cached_tokens') != 0:
                        raise RuntimeError('prompt echo, length, or cold-cache gate failed')
                    if meta.get('spec_verify_ct',0) not in (0,None) or meta.get('retraction_count',0) not in (0,None):
                        raise RuntimeError('unexpected MTP or retraction')
                    lcp = next((i for i,(x,y) in enumerate(zip(out,reference)) if x!=y),min(len(out),len(reference)))
                    row = {'trial':trial, 'output_tokens':len(out), 'output_ids_sha256':v.digest(out),
                        'finish_reason_type':meta.get('finish_reason',{}).get('type'), 'prompt_echo_exact':True,
                        'cached_tokens':0, 'native_reference_lcp':lcp,
                        'native_reference_match_count':sum(x==y for x,y in zip(out,reference)),
                        'native_reference_exact':out==reference, 'contains_eos':any(x in eos for x in out),
                        'contains_thinking_end':any(x in think_end for x in out),
                        'e2e_latency':meta.get('e2e_latency')}
                    arm['trials'].append(row)
                    v.write_json(arm_root/'checkpoint.json', arm)
                    del response, out, payload
                arm['repeat_stable'] = len({x['output_ids_sha256'] for x in arm['trials']}) == 1
                arm['completed_at_utc'] = v.now()
            finally:
                arm['cleanup'] = v.cleanup(server, a.job_id, 'baseline', IMAGE, revision)
                v.write_json(arm_root/'checkpoint.json', arm)
        left,right = (report['arms'][x] for x in VARIANTS)
        report['original_reproduced_native_window'] = all(x['native_reference_exact'] for x in left['trials'])
        report['arms_equal'] = left['trials'][0]['output_ids_sha256'] == right['trials'][0]['output_ids_sha256']
        report['status'] = 'completed' if left['repeat_stable'] and right['repeat_stable'] else 'unstable'
        report['completed_at_utc'] = v.now()
        v.write_json(o.root/'report.json', report)
        progress(report['status'])
    except Exception as exc:
        progress('failed', error_type=type(exc).__name__, error=str(exc))
        raise

if __name__ == '__main__':
    main()
