# SyftBox App for Netflix Viewing History 🍿

This project is a proof of concept utilizing [SyftBox](https://syftbox-documentation.openmined.org/) from [OpenMined](https://openmined.org/) to process 🔒 private data. The use case focuses on analyzing the [Netflix viewing history](https://help.netflix.com/en/node/101917) provided by users. This effort is part of the [#30DaysOfFLCode](https://info.openmined.org/30daysofflcode) initiative.

[![Join OpenMined on Slack](https://img.shields.io/badge/Join%20Us%20on-Slack-blue)](https://slack.openmined.org/)

## 🎯 Goals

The primary aim is to apply 🛡️ privacy-enhancing technologies to derive aggregate information from Netflix viewing history while safeguarding personal details. Some possible insights include (**_ideas are welcome_**):

- **Most common show viewed in the last week**
- **Viewing trends among participants**
- **Am I watching too much in comparison with others?**
- **Watching more due to sickness/injury?** [(source)](https://www.kaggle.com/code/nachoco/netflix-viewing-analysis-with-injury)

---

## Installation & Requirements
**_Tested on Linux and macOS._**


### 1. Preparing your data

Download your Netflix viewing activity as a CSV file from your Netflix account. See [How to download your Netflix viewing history](https://help.netflix.com/en/node/101917). Once downloaded, place the CSV file inside a directory called profile_0 inside the folder specified by the `OUTPUT_DIR` variable in your `.env` file (step 4). This can be any location in your system, for example your Downloads folder. Do **not** include "profile_0" in the path. So for a specified OUTPUT_DIR your folder structure should look as follows:

```bash
OUTPUT_DIR/
└── profile_0/
    └── NetflixViewingHistory.csv
```

### 2. Start SyftBox
Install and start SyftBox by running this command:

   ```bash
   curl -fsSL https://syftbox.net/install.sh | sh
   ```

Login using your email with the instructions in the terminal. This command is used anytime you wish to start SyftBox, it will automatically run the apps you install in the coming step.

### 3. Install the app on SyftBox

From a terminal, while syftbox is running, navigate to the SyftBox apps directory:
   ```bash
   cd /SyftBox/apps
   ```
Then clone the app's repository to install the app:
   ```bash
   git clone https://github.com/svenlankester/syftbox-netflix-svd
   ```

### 4. Set Up the Environment
Configure this app inside SyftBox.

1. Navigate to the `syftbox-netflix-svd` directory:
   ```bash
   cd syftbox-netflix-svd
   ```
2. Open the `.env` file in a text editor and **define at least** `OUTPUT_DIR`. This is the directory to make available your `profile_0/NetflixViewingHistory.csv` downloaded manually, if not available, a dummy file will be created. 

#### A more complete `.env` example:
   ```
   APP_NAME="syftbox-netflix-svd"                        # Mandatory
   AGGREGATOR_DATASITE="<aggregator-datasite-email>"     # Mandatory
   NETFLIX_EMAIL="<your-netflix-email@provider.com>"
   NETFLIX_PASSWORD="<your-password>"
   NETFLIX_PROFILE="<profile-name>"
   NETFLIX_CSV="NetflixViewingHistory.csv"               # Mandatory
   OUTPUT_DIR="/home/<your-username>/Downloads/"         # Mandatory
   AGGREGATOR_DATA_DIR="data/"                           # Mandatory
   SYFTBOX_ASSIGNED_PORT="<port>"
   ```

For MacOS users, you will likely not have your data in /home/.. but instead in /Users/.., make sure to use the correct path.

Now restart Syftbox (close using ``CTRL + C`` and re-run the command to start it) to do the initial training. This will let the aggregator add you to the read-permissions of the global model.

_Note: When running as participant, the initial run will give an error that Global_V.npy could not be found. This will resolve itself once the aggregator has detected your initial run and adds you to the read-permissions of its global model._

You will now need to wait for the aggregator to run the aggregation for the final step.

### 5. Perform the experiment

After the aggregator has added permissions to the global model for your user (this can take a while, as it depends on the person with the aggregator running the app) a local app will be deployed at ``localhost:<port>`` accessible through your browser. The port used for the app can be found in ``SyftBox/apps/syftbox-netflix-svd/logs/app.log`` in the first few lines as ``<timestamp> [syftbox] App Port: <port>``. Once you can see the app in your browser, for both lists (left and right columns) click on every show you would likely click on as a Netflix user. The app should give a pop-up window to confirm your click. Please only click each item in each list once. If the same item appears in both lists and you intend to click on it, please click on it in both lists.
