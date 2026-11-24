# gunicorn -c gunicorn.conf.py app:app          (small server: the PDF work happens in child processes, see runner.py)
bind = "0.0.0.0:8000"        # change to match your nginx / firewall setup
workers = 1                  # one is enough: only one PDF is processed at a time anyway
threads = 4                  # threads only wait for the child process, so they cost almost nothing
timeout = 930                # longer than JOB_TIMEOUT (170 s) in runner.py
graceful_timeout = 30
worker_tmp_dir = "/dev/shm"  # gunicorn's heartbeat file in RAM, no disk writes
max_requests = 200           # restart the (small) web worker now and then
accesslog = "-"
errorlog = "-"