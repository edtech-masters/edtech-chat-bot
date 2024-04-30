# Edtechmasters Chatbot

This project provides a Python chatbot built with Langchain, Chainlit, AWS Bedrock (Amazon Titan Embeddings G1 - Text and Anthropic's Claude LLM) to interact with users about educational resources.

## Prerequisites

* **Python 3.12.1 (or a compatible version):** This project requires a specific Python version to ensure compatibility with its dependencies. You can check your Python version by running `python --version` or `python3 --version` in your terminal. If you don't have the correct version, download it from [https://www.python.org/downloads/](https://www.python.org/downloads/).
* **`virtualenv` tool:** It helps isolate project dependencies. Install it using `pip install virtualenv` if you haven't already.

## Getting Started

This section provides instructions on how to set up the project's environment to run smoothly.

1. **Clone the Repository:**

   Before setting up the environment, clone this repository from GitHub.

   This will clone the repository into a local directory named `edtech-chat-bot`. You can change this name if you prefer.

2. **Change Directory:**

   Navigate into the project directory using the following command:

   ```bash
   cd edtech-chat-bot
   ```

## Setting Up the Environment

1. **Create a Virtual Environment:**

   1. Open your terminal and navigate to your project directory (where you cloned the repository).
   2. Create a virtual environment named `venv` (you can choose any name) using the following command:

      ```bash
      virtualenv venv
      ```

   3. Activate the virtual environment:

      * **Windows:**
        ```bash
        venv\Scripts\activate.bat
        ```
      * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

      The terminal prompt should change to indicate that the virtual environment is active (usually prefixed with the environment name).

2. **Install Dependencies**

   **(Optional) Environment Variables:**
      The project might utilize environment variables stored in a `.env` file. If a file named `.env-sample` exists, copy it to `.env` and update the values according to the instructions within the `.env-sample` file. These variables might be essential for the project's functionality.

   Now that the virtual environment is activated, install the required packages listed in the `requirements.txt` file using the following command:

   ```bash
   pip install -r requirements.txt
   ```

   This will download and install all the necessary Python packages for your project within the isolated virtual environment.

3. **Verify Installation**

   (Optional) After installation, you can try running a simple script from your project to ensure everything is set up correctly.

## Running the Chatbot

1. **Start the Bot:**

   Once you've completed the setup steps, you can start the chatbot using the following command:

   ```bash
   chainlit run chat.py --port 80
   ```

   Replace `chat.py` with the actual filename of your main chatbot script if it's named differently. This command instructs Chainlit to run the specified Python script (`chat.py`) and exposes it on port 80 (you can change the port number if needed).

   This will launch your chatbot in a web interface accessible from any web browser on your machine by visiting `http://localhost:80` (or the port you specified).

**Note:**

* Replace `venv` with the actual name you gave to your virtual environment.
* Deactivate the virtual environment when you're done working on the project by running `deactivate` in your terminal.

By following these steps, you'll have a properly configured environment to run this Python chatbot project using Python 3.12.1 (or a compatible version) and its dependencies isolated within the virtual environment. You can then launch the chatbot using the provided command.
