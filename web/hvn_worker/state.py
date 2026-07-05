"""
Torch-free location of a conversation's on-disk Hamiltonian state file.

Kept deliberately separate from ``hvn_worker.inference`` (which imports
``hvn.chat``/torch at module load) so the *web* role — the Django process that
serves the API and deletes conversations — can compute and remove a state file
without pulling the heavy GPU stack into its import path. ``inference.py``
re-exports this as ``_state_path`` (adding the save-time ``makedirs``), so there
is still a single definition of where the file lives.
"""
import os

from django.conf import settings


def state_path(conversation_id) -> str:
    """Absolute path to ``conversation_id``'s persisted Hamiltonian state JSON.

    Does not create the directory (unlike the save path in ``inference.py``) so
    it is safe to call from the delete path just to locate a file to remove.
    """
    conv_dir = os.path.join(settings.HVN_CHECKPOINT_DIR, "conversations")
    return os.path.join(conv_dir, f"{conversation_id}.json")
