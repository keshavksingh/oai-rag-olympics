FROM python:3.11.5

WORKDIR /

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000

ENTRYPOINT ["/bin/bash","./startup.sh"]

