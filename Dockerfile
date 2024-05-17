FROM python:3.9

RUN pip install fastapi scikit-learn pandas numpy
RUN pip install "uvicorn[standard]"

COPY ./src/*.py /src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--reload", "--port", "8000", "--host","0.0.0.0"]