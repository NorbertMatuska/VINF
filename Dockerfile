# Dockerfile for RT+Wiki Lucene search (PyLucene + PyArrow)

# Prebuilt PyLucene image – no compiling Lucene/JCC/JVM glue ourselves
FROM coady/pylucene:9.12.0

# Just in case: make sure we’re using the provided python binary
RUN python -c "import lucene; lucene.initVM(); print('lucene OK:', lucene.VERSION)"

# Python deps for your app
RUN pip install --no-cache-dir pyarrow

# App code
WORKDIR /app
COPY indexer_lucene.py /app/

# Data + index will come from mounted volumes
RUN mkdir -p /data/rt_lucene_index

ENV RT_WIKI_JOIN_PATH=/data/rt_wiki_join.parquet
ENV RT_INDEX_DIR=/data/rt_lucene_index

CMD ["python", "indexer_lucene.py"]
