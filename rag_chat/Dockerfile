FROM python

COPY . .

RUN pip install -r requirements.txt

# dummy entrypoint because we only want to exec into this container to run with an interactive tty
ENTRYPOINT ["tail", "-f", "/dev/null"]