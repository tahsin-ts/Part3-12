FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.1.3 && useradd -m -u 1000 app
WORKDIR /app
COPY elastic/ elastic/
USER app
ENV PYTHONUNBUFFERED=1
EXPOSE 29400
# The same image runs the coordinator ("server") and every node agent ("agent ...").
ENTRYPOINT ["python", "-m", "elastic"]
CMD ["server", "--host", "0.0.0.0", "--port", "29400", "--min-nodes", "2", "--max-nodes", "4"]
