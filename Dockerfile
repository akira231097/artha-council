# Artha Council — minimal runtime image.
#
# This is pure-Python (requests, pandas, numpy, LLM SDKs). No system media
# libraries are required. Provider keys are supplied at runtime via -e / --env-file;
# never bake credentials into the image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the application source.
COPY . .

# Sanity check: the package and entry point compile.
RUN python -m compileall artha run.py

# run.py is the CLI entry point (~30 subcommands). Default to the help text;
# pass a subcommand at run time, e.g. `docker run ... overview`.
ENTRYPOINT ["python", "run.py"]
CMD ["overview"]
