import json

import pysftp as pysftp


def sftp_connection():
    with open('../_resources/config.json') as f:
        data = json.load(f)
        host = data['host']
        user = data['user']
        key = data['key']

    print("Connecting to remote server")
    return pysftp.Connection(host, username=user, private_key=key, cnopts=pysftp.CnOpts())
