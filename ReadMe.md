### Running the Chain

To run the chain, use Python’s `-m` flag from the `lang_chain_basics` directory:

```bash
python -m chains.1_chain
```

The `-m` flag tells Python to treat the target as a module within a package, allowing you to use relative imports like:

```python
from ..starter import llm
```

**Tip:** Alternatively, you can copy `starter.py` into every module and remove the relative import. This works but may result in duplicate files.

