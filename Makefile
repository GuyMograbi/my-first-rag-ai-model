.PHONY: install run clean

install:
	@pip install -r requirements.txt

run:
	@python chatbot.py

clean:
	@rm -rf __pycache__