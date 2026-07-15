# Single image for the whole Django/Celery/Channels app. The same image runs
# as different roles (web / worker / beat) via docker-entrypoint.sh + $ROLE -
# see ISSUE_30_PLAN.md "Session Decisions" #3. Postgres/Redis are NOT in this
# image; they're separate official images in docker-compose.yml / docker-prod.yml.
#
# Build context is the REPO ROOT (not web/), so this image can also vendor the
# existing hvn/ package that the worker role needs to load the model:
#   docker build -f web/Dockerfile -t kramba-web .
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=web.settings.prod \
    PYTHONPATH=/srv

WORKDIR /srv

# hvn/ dependencies (torch is already in the base image; root requirements.txt
# pins triton-windows which is Windows-only, so this container uses a
# Linux-appropriate subset instead - see hvn-requirements-docker.txt)
COPY web/hvn-requirements-docker.txt /srv/hvn-requirements.txt
RUN pip install --no-cache-dir -r /srv/hvn-requirements.txt

# Django/Celery/Channels dependencies
COPY web/requirements.txt /srv/web-requirements.txt
RUN pip install --no-cache-dir -r /srv/web-requirements.txt

# Existing HVN architecture package (hvn/core, hvn/chat.py, hvn/server.py, ...).
# train.py is vendored too because CharTokenizer lives there (not under hvn/) -
# hvn_worker/worker.py reuses it rather than duplicating a tokenizer class.
COPY hvn/ /srv/hvn/
COPY train.py /srv/train.py
# EdgeHVN serving (issue #97 Phase B): hvn_worker/worker.py's edgehvn branch
# reuses load_checkpoint_for_resume (block_plan-aware hybrid checkpoint
# loading) rather than duplicating that construction logic. Only this one
# file is needed -- it has no dependency on other root-level training
# scripts (verified: imports only hvn.*, torch, stdlib).
COPY train_edge_conversational.py /srv/train_edge_conversational.py
# In-app training (Issue #57): the training worker reuses auto_train.py's
# curriculum orchestrator and the eval/ dataset+scoring package; the web role
# also imports eval.curriculum (torch-free) for the CURRICULUM_STAGES the launch
# form validates against. Same image, so vendor both here.
COPY auto_train.py /srv/auto_train.py
COPY eval/ /srv/eval/

# Django project
COPY web/ /srv/web/
WORKDIR /srv/web

RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /srv \
    && mkdir -p /data/checkpoints \
    && chown -R appuser:appuser /data/checkpoints
USER appuser

# Invoke through `sh` rather than execing the script directly, so a stray CRLF
# shebang (#!/bin/sh\r from a Windows checkout) can't cause
# "exec .../docker-entrypoint.sh: no such file or directory". The .gitattributes
# eol=lf rule is the primary fix; this is belt-and-suspenders (issue #58).
ENTRYPOINT ["sh", "/srv/web/docker-entrypoint.sh"]
CMD ["web"]
