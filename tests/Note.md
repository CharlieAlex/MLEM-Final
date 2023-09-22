1. Do Not add .. when importing.
    - e.g. `from etl_func.cut_text_aux import *`
2. When running the python code, run on the parent dir.
    - i.e. `cdml`
3. Start running the test.
    1. Use . to note the relative path when running.
        - e.g. `python3 -m tests.test_cut_text_aux`
    2. BUT!! Normally, use `pytest -vv` is enough!!
