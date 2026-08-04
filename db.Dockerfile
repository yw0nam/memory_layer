FROM pgvector/pgvector:pg17

ADD https://github.com/timescale/pg_textsearch/releases/download/v1.3.1/pg-textsearch-v1.3.1-pg17-amd64.zip /tmp/pg_textsearch.zip
RUN apt-get update \
    && apt-get install -y --no-install-recommends unzip \
    && unzip /tmp/pg_textsearch.zip -d /tmp \
    && dpkg -i /tmp/pg-textsearch-postgresql-17_1.3.1-1_amd64.deb \
    && apt-get purge -y unzip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /tmp/pg_textsearch.zip /tmp/pg-textsearch-postgresql-17_1.3.1-1_amd64.deb
