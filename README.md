## 1. Add dependency

```bash
poetry add <dependency>
```

 it:
* updates `pyproject.toml`
* updates `poetry.lock`
* installs the package


---

## 3. `poetry install`

```bash
poetry install
```

* Installs dependencies from `poetry.lock`
* Creates virtual environment if needed
* Installs your project (unless disabled)

Used when:

* you clone a repo
* or recreate environment

---

## The summary steps

```bash
# Add dependency (also installs + updates lock)
poetry add <dependency>

# Reinstall environment from lock file
poetry install

# Run commands without activating
poetry run python script.py
```