import os 
import pickle

def save_stub(stub_path, object):
    if stub_path is None:
        return

    stub_dir = os.path.dirname(stub_path)
    if stub_dir:
        os.makedirs(stub_dir, exist_ok=True)

    with open(stub_path, 'wb') as f:
        pickle.dump(object, f)

def load_stub(read_from_stub, stub_path):
    if read_from_stub is True and stub_path is not None and os.path.exists(stub_path):
        with open(stub_path, 'rb') as f:
            return pickle.load(f)
    return None