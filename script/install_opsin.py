"""Download official OPSIN, verify SHA256, compile a local status-aware bridge."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PINNED_VERSION = '2.9.0'
PINNED_SHA256 = 'c2e29326c281f87b59a05d934d8589adac6e9d17b95b984931b3e739111b360f'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--version', default=PINNED_VERSION, help='Pinned tested release or latest')
    ap.add_argument('--directory', type=Path, default=ROOT/'tools/opsin')
    ap.add_argument('--jar', type=Path, help='Use an already downloaded official JAR')
    ap.add_argument('--javac', default='javac')
    ap.add_argument('--no-proxy', action='store_true', help='Ignore environment HTTP proxies when downloading')
    args = ap.parse_args()
    if not shutil.which(args.javac):
        ap.error('javac not found; install a Java JDK 8 or newer')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.no_proxy else urllib.request.build_opener()
    def download(url):
        request = urllib.request.Request(url, headers={'User-Agent': 'SUPERChem-OPSIN-installer'})
        with opener.open(request, timeout=120) as response:
            return response.read()
    version = args.version
    if version == PINNED_VERSION:
        filename = f'opsin-cli-{version}-jar-with-dependencies.jar'
        url = f'https://github.com/dan2097/opsin/releases/download/{version}/{filename}'
        expected = PINNED_SHA256
    else:
        suffix = 'latest' if version == 'latest' else 'tags/'+version
        release = json.loads(download('https://api.github.com/repos/dan2097/opsin/releases/'+suffix))
        version = release['tag_name']
        asset = next(a for a in release['assets'] if a['name'].startswith('opsin-cli-') and a['name'].endswith('-jar-with-dependencies.jar'))
        filename, url = asset['name'], asset['browser_download_url']
        expected = (asset.get('digest') or '').removeprefix('sha256:')
        if len(expected) != 64:
            raise ValueError('Release has no SHA256 asset digest; cannot verify automatically')
    folder = args.directory.resolve(); folder.mkdir(parents=True, exist_ok=True)
    target = folder/filename
    if not target.exists():
        payload = args.jar.read_bytes() if args.jar else download(url)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('OPSIN JAR SHA256 mismatch')
        with tempfile.NamedTemporaryFile(dir=folder, suffix='.jar.tmp', delete=False) as handle:
            handle.write(payload); temporary = Path(handle.name)
        temporary.replace(target)
    if digest(target) != expected:
        raise ValueError('Existing JAR differs from the verified official release; refusing overwrite')
    bridge = ROOT/'superchem/resources/OpsinBridge.java'
    classes = folder/'classes'; classes.mkdir(exist_ok=True)
    subprocess.run([args.javac, '-encoding', 'UTF-8', '-cp', str(target), '-d', str(classes), str(bridge)], check=True)
    # Keep original third-party licensing available next to the binary.
    if not (folder/'OPSIN-LICENSE.txt').exists():
        license_url = f'https://raw.githubusercontent.com/dan2097/opsin/{version}/LICENSE.txt'
        if not args.jar:
            (folder/'OPSIN-LICENSE.txt').write_bytes(download(license_url))
    metadata = {'version': version, 'release_url': 'https://github.com/dan2097/opsin/releases/tag/'+version,
                'jar': filename, 'jar_sha256': expected, 'bridge_source_sha256': digest(bridge),
                'bridge_class_sha256': digest(classes/'OpsinBridge.class'), 'license': 'OPSIN MIT; JAR bundles third-party dependencies'}
    (folder/'manifest.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__': main()
