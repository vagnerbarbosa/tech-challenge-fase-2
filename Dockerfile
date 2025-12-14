FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install --only-binary=:all: -r requirements.txt
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]