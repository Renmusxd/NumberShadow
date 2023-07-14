wheel:
	maturin build --release --strip

.PHONY: clean
clean:
	cargo clean
