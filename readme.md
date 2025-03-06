
### Files and Directories

- `main.py`: The main script that runs the trading agent.
- `utils/hyperliquid_util.py`: Utility functions for setting up and interacting with the Hyperliquid API.
- `.vscode/launch.json`: Configuration for debugging in Visual Studio Code.
- `requirements.txt`: List of Python dependencies.
- `setup.bat`: Batch script to set up the Python virtual environment.
- `.env`: Environment variables (not included in version control).
- `.gitignore`: Specifies files and directories to be ignored by Git.

## Setup

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd hyperliquid_trading_agent
    ```

2. **Set up the virtual environment:**

    ```sh
    setup.bat
    ```

3. **Install dependencies:**

    ```sh
    .venv\Scripts\activate
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a [.env](http://_vscodecontentref_/7) file in the root directory and add the following variables:

    ```env
    NEWS_API_KEY=your_news_api_key
    OPENAI_API_KEY=your_openai_api_key
    HL_SECRET_KEY=your_hyperliquid_secret_key
    HL_ACCOUNT_ADDRESS=your_hyperliquid_account_address
    ```

## Usage

To run the trading agent, execute the [main.py](http://_vscodecontentref_/8) script:

```sh
python main.py