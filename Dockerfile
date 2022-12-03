FROM debian:bullseye

RUN apt-get update
RUN apt-get install -y r-base python3 python3-pip r-cran-devtools git
RUN Rscript -e "devtools::install_github('DE0CH/irace')"
COPY requirements.txt .
RUN pip3 install -r requirements.txt