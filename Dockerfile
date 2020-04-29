#I specify the parent base image which is the python version 3.7
FROM python:3.6

MAINTAINER  eoderinde <oderinde.taiwo@idaf.ng>

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# install system dependencies
RUN apt-get update \
    && apt-get -y install gcc make \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
RUN pip install --no-cache-dir --upgrade pip

# set work directory
WORKDIR /src/app

# copy requirements.txt
COPY ./requirements.txt /src/app/requirements.txt

# install project requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# Generate pikle file
WORKDIR /src/app/terragon_model
RUN python model.py

# set work directory
WORKDIR /src/app

# set app port
EXPOSE 8080

ENTRYPOINT [ "python" ] 

# Run app.py when the container launches
CMD [ "app.py","run","--host","0.0.0.0"]


cd AKIN\terragon

docker build eoderinde/terragon .

docker build -t terragon_model:1.0 .
# or
docker build -t eoderinde/terragon_model:1.0 .

docker images

docker run --name deployML -p 8080:8080 eoderinde/terragon_model:1.0

docker ps 

