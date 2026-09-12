FROM python:3.12-slim

# Compiling/building dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd --create-home --shell /bin/bash thesis-user
WORKDIR /home/thesis-user/projects/thesis
RUN chown -R thesis-user:thesis-user /home/thesis-user
USER thesis-user

# Python dependencies
COPY --chown=thesis-user:thesis-user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
ENV PATH="/home/thesis-user/.local/bin:${PATH}"

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
