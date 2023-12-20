

import sys
print(sys.path)
print(sys.executable)

import subprocess
subprocess.check_call(['cat', '/usr/local/lib/python3.10/io.py'])
#
# from _io import text_encoding
# print("_io Encoding: " + text_encoding(None))
#
# from io import text_encoding
# print("io Encoding: " + text_encoding(None))


import subprocess
subprocess.check_call(['chmod', '1777', '/tmp'])
subprocess.check_call(['apt-get', 'update'])
subprocess.check_call(['apt-mark', 'hold', 'python3-distro'])
subprocess.check_call(['apt-mark', 'hold', 'ssh-import-id'])
# subprocess.check_call(['apt-get', 'install', '-y', 'openssh-server'])
subprocess.check_call(['apt-get', 'install', '-y', 'ssh', 'net-tools', 'procps', 'less', 'jq', 'vim', 'rsync', 'locales', 'rsyslog'])

