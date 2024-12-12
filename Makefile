SRC=main.py

all: default

default:
	@echo "start"
	@echo "----------------------------------------"
	@python $(SRC)
	@echo "----------------------------------------"
	@echo "end";