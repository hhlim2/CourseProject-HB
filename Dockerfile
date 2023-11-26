<<<<<<< HEAD
# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /CourseProject-HB


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


COPY . .

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT [ "streamlit", "run" , "final_project.py",  "--server.port=8080", "--server.address=0.0.0.0"]







=======
# syntax=docker/dockerfile:1

FROM python:3.10-slim-buster

WORKDIR /to_deploy


COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


COPY . .

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT [ "streamlit", "run" , "final_project.py",  "--server.port=8080", "--server.address=0.0.0.0"]







>>>>>>> ab8be327ee478b301cf7a52630cda2e2750b2610
