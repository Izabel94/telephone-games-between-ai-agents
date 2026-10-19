# Telephone Games Between AI Agents

Workshop notebook: a chain of AI agents passes a person's personality description along, and we measure what gets lost and whether the person turns into a stereotype.

## What you need

- A laptop (Mac or Windows) and an internet connection. Python is optional: if Python 3.12, 3.13 or 3.14 is not installed, setup installs Python 3.12 inside the `tw` folder.
- A web browser.
- **No code editor and no typing commands.** You double-click two files: `Setup-Windows` (or `Setup-Mac`) once, then `Start-Windows` (or `Start-Mac`).

## Before the workshop

1. **Get a free Gemini API key** at [aistudio.google.com](https://aistudio.google.com) with a personal Google account (18+): click "Get API key", then "Create API key", and copy it. University or company accounts may have AI Studio switched off.
2. **Unzip `tw.zip`** (Windows: right-click, Extract All) and open the `tw` folder. **On Windows, unzip it into your user folder** (type `%USERPROFILE%` in the File Explorer address bar), **not into OneDrive or a deep folder**: Windows cannot install files with very long paths. Setup checks this and tells you if the folder is too deep.
3. **Double-click `Setup-Windows`** (on a Mac: `Setup-Mac`). A window opens and:
   - finds Python, or installs it inside the `tw` folder if it is missing or too old (nothing else on your laptop changes),
   - installs everything (the first time this takes a while),
   - asks for your key: **paste it** (Windows: right-click or Ctrl+V, Mac: Cmd+V) and press Enter,
   - checks the models answer. Wait until it says **"Setup done"**.
4. **Double-click `Start-Windows`** (on a Mac: `Start-Mac`). The notebook opens in your browser. **Keep the black window open** while you work.

**If your computer blocks the file** (it came from the internet):
- Windows says "Windows protected your PC": click **More info**, then **Run anyway**.
- Mac says it "cannot be opened": open **System Settings → Privacy & Security**, scroll down and click **Open Anyway** next to the file's name. (On older macOS: right-click the file → Open → Open.)

Next time, just double-click `Start-Windows` (or `Start-Mac`).

## Using the notebook

- The start file opens the notebook **in your browser**. If no tab opens, copy the link starting with `http://localhost:8888/lab?token=` from the black window into your browser.
- **Keep the black window open** while you work. Closing it stops the notebook.
- A notebook is a page of **cells**. Click a code cell and press **Shift + Enter** to run it. Run cells from top to bottom.
- The live cells call the AI. Measured on Gemma 26B: "Part 1" took 22 s, "Your turn" (2 chains) 46 s; slower Gemini models take longer (see the table in the notebook). Everything in "Part 3" uses the preloaded results and appears at once.
- To stop: close the browser tab, then close the black window.
- Already use VS Code? You can open `workshop.ipynb` there instead, if the Python and Jupyter extensions are installed: pick the `.venv` folder as the kernel. Not needed for the workshop.

## Files

| File | What it is |
|---|---|
| `Setup-Windows.bat` / `Setup-Mac.command` | Double-click once: installs everything and saves your key |
| `Start-Windows.bat` / `Start-Mac.command` | Double-click to open the notebook |
| `workshop.ipynb` | The notebook: a live chain, your own experiment, and charts of the big run |
| `telephone.py` | Profiles, prompts, the chain, rate limiting, saving. `python telephone.py` runs a self-check |
| `analysis.py` | Tables and charts |
| `check_models.py` | Tests your key: one call per model with the real settings |
| `results/` | The big run (180 chains), **preloaded**: it took about an hour, so you don't need to run it |

## Reusing the code for research

The notebook ends with two sections for this: **Export the data** (CSV files for R, SPSS or Excel, one row per agent with the full text and scores) and **Reuse this for your own study** (where to change the questionnaire, the people, the prompts, the models and the chain settings). The **Method details** section near the top shows the exact prompts, the measures and their chance levels.

Every new run saves, per agent, the model asked for, the model version that actually answered, the time (UTC) and a seed per repeat. The preloaded run predates this, so those fields are empty there.

## If double-click does not work

The same steps by hand, in a terminal:

1. **Open a terminal in this folder and install.** Open the **Terminal** app (Mac) or **PowerShell** (Windows), type `cd ` (with a space), drag the folder that contains `README.md` into the window, and press Enter.

   **Check you are in the right place:** type `ls` (Windows: `dir`). You must see `README.md`, `workshop.ipynb` and `requirements.txt`. If not, you are in the wrong folder: all following commands will fail.

   Mac / Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Windows (Command Prompt):
   ```
   py -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   Windows (PowerShell): same, but activate with `.venv\Scripts\Activate.ps1`. If PowerShell says scripts are disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.
2. **Add your key.** `.env.example` is a hidden file on Mac, so copy it with a command:
   - Mac / Linux: `cp .env.example .env`
   - Windows: `copy .env.example .env`

   Then open `.env` in any text editor and replace `paste-your-key-here` with your key. Never share this file.
3. **Check it works:** `python check_models.py`. Every line should say "ok".
4. **Open the notebook:** `jupyter lab workshop.ipynb`

> Every time you open a new terminal, activate first (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), then run `jupyter lab workshop.ipynb`.

## Free-tier notes

- Gemma 4 31B failed most calls on the free tier in our tests, so it is left out.
- Limits are **per Google Cloud project**, not per key. Gemma 4 has a large daily limit; Gemini Flash models are small. The notebook shows how many calls a run needs before you start.
- On the free tier, Google may use your prompts to improve its products. Fine for made-up profiles; do not send real participant data.
