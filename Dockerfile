# 300MB ~ real	1m32.339s

# Few Issues:
# WARNING: apt does not have a stable CLI interface. Use with caution in scripts.
# Also says "Notebook project/20190602_All-Chapters.ipynb is not trusted"

# TO DO
# Use requirements file pip3 -r requirements 
# May have to install blas, mkl and other packages later on 
# ssh a IDE into the container 


#FROM python:3.8.2-slim-buster
FROM python:3.8.6-slim-buster

#CMD ["bin/sh"]

RUN apt-get update && \
	apt-get install -y && \
    pip3  --no-cache-dir install jupyter numpy pandas scikit-learn matplotlib seaborn kaggle

EXPOSE 8889

# So that the project does not start in the root directory
WORKDIR /project-home 

# Optional
VOLUME /project-home

CMD ["jupyter", "notebook", "--ip='*'", "--port=8889", "--no-browser", "--allow-root"]

# Next Steps :

# docker build -t banktermdeposit . 
# sh temp.sh

