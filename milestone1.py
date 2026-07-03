{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e5764f55",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:23.352300Z",
     "iopub.status.busy": "2026-07-03T14:44:23.352025Z",
     "iopub.status.idle": "2026-07-03T14:44:25.114810Z",
     "shell.execute_reply": "2026-07-03T14:44:25.114147Z"
    },
    "papermill": {
     "duration": 1.768154,
     "end_time": "2026-07-03T14:44:25.116462+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:23.348308+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/competitions/smart-mcq-solver-challenge/sample_submission.csv\n",
      "/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv\n",
      "/kaggle/input/competitions/smart-mcq-solver-challenge/test.csv\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session\n",
    "\n",
    "# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session\n",
    "# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md\n",
    "\n",
    "import kagglehub\n",
    "# kagglehub.dataset_download('<owner>/<dataset-slug>')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "86fbc4e2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:25.122045Z",
     "iopub.status.busy": "2026-07-03T14:44:25.121633Z",
     "iopub.status.idle": "2026-07-03T14:44:25.199407Z",
     "shell.execute_reply": "2026-07-03T14:44:25.198660Z"
    },
    "papermill": {
     "duration": 0.082236,
     "end_time": "2026-07-03T14:44:25.200757+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:25.118521+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>prompt</th>\n",
       "      <th>A</th>\n",
       "      <th>B</th>\n",
       "      <th>C</th>\n",
       "      <th>D</th>\n",
       "      <th>E</th>\n",
       "      <th>answer</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Pick the best possible answer: What is Martin ...</td>\n",
       "      <td>Martin Heidegger believes that humans exist wi...</td>\n",
       "      <td>Martin Heidegger believes that humans do not e...</td>\n",
       "      <td>Martin Heidegger does not believe in the exist...</td>\n",
       "      <td>Martin Heidegger believes that the relationshi...</td>\n",
       "      <td>Martin Heidegger believes that time is an illu...</td>\n",
       "      <td>B</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>What is accelerator-based light-ion fusion?</td>\n",
       "      <td>Accelerator-based light-ion fusion is a techni...</td>\n",
       "      <td>Accelerator-based light-ion fusion is a techni...</td>\n",
       "      <td>Accelerator-based light-ion fusion is a techni...</td>\n",
       "      <td>Accelerator-based light-ion fusion is a techni...</td>\n",
       "      <td>Accelerator-based light-ion fusion is a techni...</td>\n",
       "      <td>A</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Determine the correct option: What is the term...</td>\n",
       "      <td>Blueshifting</td>\n",
       "      <td>Redshifting</td>\n",
       "      <td>Reddening</td>\n",
       "      <td>Whitening</td>\n",
       "      <td>Yellowing</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Select the most accurate option: What is Marti...</td>\n",
       "      <td>Martin Heidegger believes that humans exist wi...</td>\n",
       "      <td>Martin Heidegger believes that humans do not e...</td>\n",
       "      <td>Martin Heidegger does not believe in the exist...</td>\n",
       "      <td>Martin Heidegger believes that the relationshi...</td>\n",
       "      <td>Martin Heidegger believes that time is an illu...</td>\n",
       "      <td>B</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Identify the correct statement: What is the co...</td>\n",
       "      <td>Simultaneity is relative, meaning that two eve...</td>\n",
       "      <td>Simultaneity is relative, meaning that two eve...</td>\n",
       "      <td>Simultaneity is absolute, meaning that two eve...</td>\n",
       "      <td>Simultaneity is a concept that applies only to...</td>\n",
       "      <td>Simultaneity is a concept that applies only to...</td>\n",
       "      <td>A</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id                                             prompt  \\\n",
       "0   1  Pick the best possible answer: What is Martin ...   \n",
       "1   2        What is accelerator-based light-ion fusion?   \n",
       "2   3  Determine the correct option: What is the term...   \n",
       "3   4  Select the most accurate option: What is Marti...   \n",
       "4   5  Identify the correct statement: What is the co...   \n",
       "\n",
       "                                                   A  \\\n",
       "0  Martin Heidegger believes that humans exist wi...   \n",
       "1  Accelerator-based light-ion fusion is a techni...   \n",
       "2                                       Blueshifting   \n",
       "3  Martin Heidegger believes that humans exist wi...   \n",
       "4  Simultaneity is relative, meaning that two eve...   \n",
       "\n",
       "                                                   B  \\\n",
       "0  Martin Heidegger believes that humans do not e...   \n",
       "1  Accelerator-based light-ion fusion is a techni...   \n",
       "2                                        Redshifting   \n",
       "3  Martin Heidegger believes that humans do not e...   \n",
       "4  Simultaneity is relative, meaning that two eve...   \n",
       "\n",
       "                                                   C  \\\n",
       "0  Martin Heidegger does not believe in the exist...   \n",
       "1  Accelerator-based light-ion fusion is a techni...   \n",
       "2                                          Reddening   \n",
       "3  Martin Heidegger does not believe in the exist...   \n",
       "4  Simultaneity is absolute, meaning that two eve...   \n",
       "\n",
       "                                                   D  \\\n",
       "0  Martin Heidegger believes that the relationshi...   \n",
       "1  Accelerator-based light-ion fusion is a techni...   \n",
       "2                                          Whitening   \n",
       "3  Martin Heidegger believes that the relationshi...   \n",
       "4  Simultaneity is a concept that applies only to...   \n",
       "\n",
       "                                                   E answer  \n",
       "0  Martin Heidegger believes that time is an illu...      B  \n",
       "1  Accelerator-based light-ion fusion is a techni...      A  \n",
       "2                                          Yellowing      C  \n",
       "3  Martin Heidegger believes that time is an illu...      B  \n",
       "4  Simultaneity is a concept that applies only to...      A  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('/kaggle/input/competitions/smart-mcq-solver-challenge/train.csv')\n",
    "df.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "a7df7037",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:25.205933Z",
     "iopub.status.busy": "2026-07-03T14:44:25.205648Z",
     "iopub.status.idle": "2026-07-03T14:44:25.218504Z",
     "shell.execute_reply": "2026-07-03T14:44:25.217788Z"
    },
    "papermill": {
     "duration": 0.016921,
     "end_time": "2026-07-03T14:44:25.219750+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:25.202829+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "answer\n",
      "B    490\n",
      "C    459\n",
      "A    369\n",
      "D    358\n",
      "E    324\n",
      "Name: count, dtype: int64\n",
      "Sum of the most frequent and least frequent option is : 814\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 1 814\n",
    "distribution = df['answer'].value_counts()\n",
    "print(distribution)\n",
    "summflf = distribution.max() + distribution.min()\n",
    "print(\"Sum of the most frequent and least frequent option is :\", summflf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d20b8738",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:25.225455Z",
     "iopub.status.busy": "2026-07-03T14:44:25.224592Z",
     "iopub.status.idle": "2026-07-03T14:44:25.241735Z",
     "shell.execute_reply": "2026-07-03T14:44:25.240906Z"
    },
    "papermill": {
     "duration": 0.02131,
     "end_time": "2026-07-03T14:44:25.243114+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:25.221804+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Vocabulory size (unique words): 859\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 2 859\n",
    "import string \n",
    "\n",
    "df['cleaned_prompt'] = df['prompt'].astype(str).str.lower()\n",
    "df['cleaned_prompt'] = df['cleaned_prompt'].str.translate(\n",
    "    str.maketrans('','',string.punctuation)\n",
    ")\n",
    "\n",
    "all_words=[]\n",
    "for text in df['cleaned_prompt']:\n",
    "    words=text.split()\n",
    "    all_words.extend(words)\n",
    "\n",
    "unique_words = set(all_words)\n",
    "print(\"Vocabulory size (unique words):\", len(unique_words))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "bd3d4be6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:25.248139Z",
     "iopub.status.busy": "2026-07-03T14:44:25.247882Z",
     "iopub.status.idle": "2026-07-03T14:44:26.933271Z",
     "shell.execute_reply": "2026-07-03T14:44:26.932519Z"
    },
    "papermill": {
     "duration": 1.689451,
     "end_time": "2026-07-03T14:44:26.934645+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:25.245194+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cleaned prompt: pick the best possible answer what is martin heideggers view on the relationship between time and human existence among the listed options\n",
      "Words before filtering: ['pick', 'the', 'best', 'possible', 'answer', 'what', 'is', 'martin', 'heideggers', 'view', 'on', 'the', 'relationship', 'between', 'time', 'and', 'human', 'existence', 'among', 'the', 'listed', 'options']\n",
      "Count before: 22\n",
      "Words after filtering: ['pick', 'best', 'possible', 'answer', 'martin', 'heideggers', 'view', 'relationship', 'time', 'human', 'existence', 'listed', 'options']\n",
      "Count after: 13\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 3 13\n",
    "\n",
    "from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS\n",
    "\n",
    "prompt_row1 = df['cleaned_prompt'].iloc[0]\n",
    "print(\"Cleaned prompt:\", prompt_row1)\n",
    "\n",
    "words = prompt_row1.split()\n",
    "print(\"Words before filtering:\", words)\n",
    "print(\"Count before:\", len(words))\n",
    "\n",
    "filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]\n",
    "print(\"Words after filtering:\", filtered_words)\n",
    "print(\"Count after:\", len(filtered_words))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "1e96684a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:26.941977Z",
     "iopub.status.busy": "2026-07-03T14:44:26.941532Z",
     "iopub.status.idle": "2026-07-03T14:44:27.099628Z",
     "shell.execute_reply": "2026-07-03T14:44:27.098939Z"
    },
    "papermill": {
     "duration": 0.162959,
     "end_time": "2026-07-03T14:44:27.101097+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:26.938138+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total number of feature columns (vocabulary size): 2762\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 4  2762\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "combined_text = (\n",
    "    df['prompt'].tolist() +\n",
    "    df['A'].tolist() +\n",
    "    df['B'].tolist() +\n",
    "    df['C'].tolist() +\n",
    "    df['D'].tolist() +\n",
    "    df['E'].tolist()\n",
    ")\n",
    "\n",
    "vectorizer = TfidfVectorizer(stop_words='english')\n",
    "vectorizer.fit(combined_text)\n",
    "\n",
    "vocab_size = len(vectorizer.vocabulary_)\n",
    "print(\"Total number of feature columns (vocabulary size):\", vocab_size)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5a4246db",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:27.106588Z",
     "iopub.status.busy": "2026-07-03T14:44:27.106312Z",
     "iopub.status.idle": "2026-07-03T14:44:27.225279Z",
     "shell.execute_reply": "2026-07-03T14:44:27.224126Z"
    },
    "papermill": {
     "duration": 0.123596,
     "end_time": "2026-07-03T14:44:27.226938+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:27.103342+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Prompt: Pick the best possible answer: What is Martin Heidegger's view on the relationship between time and human existence? among the listed options.\n",
      "Option A: Martin Heidegger believes that humans exist within a time continuum that is infinite and does not have a defined beginning or end. The relationship to the past involves acknowledging it as a historical era, and the relationship to the future involves creating a world that will endure beyond one's own time.\n",
      "Cosine Similarity: 0.2328\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 5  0.2328\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "prompt_row1 = df['prompt'].iloc[0]\n",
    "optionA_row1 = df['A'].iloc[0]\n",
    "\n",
    "print(\"Prompt:\", prompt_row1)\n",
    "print(\"Option A:\", optionA_row1)\n",
    "\n",
    "prompt_vector = vectorizer.transform([prompt_row1])\n",
    "optionA_vector = vectorizer.transform([optionA_row1])\n",
    "\n",
    "similarity = cosine_similarity(prompt_vector, optionA_vector)\n",
    "\n",
    "print(\"Cosine Similarity:\", round(similarity[0][0], 4))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ea7483f9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:27.232857Z",
     "iopub.status.busy": "2026-07-03T14:44:27.232410Z",
     "iopub.status.idle": "2026-07-03T14:44:27.481576Z",
     "shell.execute_reply": "2026-07-03T14:44:27.480456Z"
    },
    "papermill": {
     "duration": 0.254153,
     "end_time": "2026-07-03T14:44:27.483289+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:27.229136+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 13.7%\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 6 13.7%\n",
    "options = ['A', 'B', 'C', 'D', 'E']\n",
    "prompt_vectors = vectorizer.transform(df['prompt'])\n",
    "\n",
    "similarity_scores = {}\n",
    "for option in options:\n",
    "    option_vectors = vectorizer.transform(df[option])\n",
    "    # diagonal gives similarity of each row's prompt with its own option\n",
    "    similarity_scores[option] = cosine_similarity(prompt_vectors, option_vectors).diagonal()\n",
    "\n",
    "scores_df = pd.DataFrame(similarity_scores)\n",
    "predicted = scores_df.idxmax(axis=1) \n",
    "\n",
    "accuracy = (predicted == df['answer']).mean() * 100\n",
    "print(f\"Accuracy: {round(accuracy, 2)}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a757d134",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:27.488742Z",
     "iopub.status.busy": "2026-07-03T14:44:27.488491Z",
     "iopub.status.idle": "2026-07-03T14:44:27.494586Z",
     "shell.execute_reply": "2026-07-03T14:44:27.493508Z"
    },
    "papermill": {
     "duration": 0.010415,
     "end_time": "2026-07-03T14:44:27.495928+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:27.485513+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "MAP@3 score: 1.0\n",
      "MAP@3 score: 0.5\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 7 MAP@3 score: 1.0 MAP@3 score: 0.5\n",
    "def apk(actual, predicted, k=3):\n",
    "    score = 0.0\n",
    "    num_hits = 0.0\n",
    "    for i, pred in enumerate(predicted[:k]):\n",
    "        if pred == actual:\n",
    "            num_hits += 1\n",
    "            score += num_hits / (i + 1)\n",
    "    return score / min(k, 1)\n",
    "\n",
    "actual = 'C'\n",
    "predicted = ['C', 'A', 'B']\n",
    "print(f\"MAP@3 score: {apk(actual, predicted)}\")\n",
    "\n",
    "#ANSWER 8 \n",
    "actual = 'B'\n",
    "predicted = ['D', 'B', 'E']\n",
    "print(f\"MAP@3 score: {apk(actual, predicted)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "09cb8854",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:27.501771Z",
     "iopub.status.busy": "2026-07-03T14:44:27.501507Z",
     "iopub.status.idle": "2026-07-03T14:44:27.507844Z",
     "shell.execute_reply": "2026-07-03T14:44:27.507007Z"
    },
    "papermill": {
     "duration": 0.010878,
     "end_time": "2026-07-03T14:44:27.509182+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:27.498304+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Majority Class Baseline MAP@3: 0.4213\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 9 Majority Class Baseline MAP@3: 0.4213\n",
    "static_prediction = ['B', 'C', 'A']  \n",
    "\n",
    "scores = []\n",
    "for actual in df['answer']:\n",
    "    score = apk(actual, static_prediction, k=3)\n",
    "    scores.append(score)\n",
    "\n",
    "map3_score = sum(scores) / len(scores)\n",
    "print(f\"Majority Class Baseline MAP@3: {round(map3_score, 4)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "027693e6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-07-03T14:44:27.514967Z",
     "iopub.status.busy": "2026-07-03T14:44:27.514718Z",
     "iopub.status.idle": "2026-07-03T14:44:28.249467Z",
     "shell.execute_reply": "2026-07-03T14:44:28.248561Z"
    },
    "papermill": {
     "duration": 0.739595,
     "end_time": "2026-07-03T14:44:28.251094+00:00",
     "exception": false,
     "start_time": "2026-07-03T14:44:27.511499+00:00",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "          A         B         C         D         E\n",
      "0  0.232800  0.258421  0.498558  0.451256  0.169300\n",
      "1  0.501327  0.501201  0.528616  0.473393  0.374950\n",
      "2  0.000000  0.000000  0.000000  0.000000  0.000000\n",
      "3  0.247120  0.274316  0.529224  0.479013  0.179714\n",
      "4  0.071975  0.073314  0.072467  0.209584  0.209584\n",
      "Sample predictions: ['C', 'D', 'B']\n",
      "TF-IDF Pipeline MAP@3: 0.3119\n"
     ]
    }
   ],
   "source": [
    "#ANSWER 10  TF-IDF Pipeline MAP@3: 0.3119\n",
    "\n",
    "print(scores_df.head())\n",
    "\n",
    "top3_predictions = scores_df.apply(\n",
    "    lambda row: row.nlargest(3).index.tolist(), axis=1\n",
    ")\n",
    "print(\"Sample predictions:\", top3_predictions.iloc[0])\n",
    "\n",
    "scores = []\n",
    "for i in range(len(df)):\n",
    "    actual = df['answer'].iloc[i]\n",
    "    predicted = top3_predictions.iloc[i]\n",
    "    scores.append(apk(actual, predicted, k=3))\n",
    "\n",
    "map3_score = sum(scores) / len(scores)\n",
    "print(f\"TF-IDF Pipeline MAP@3: {round(map3_score, 4)}\")"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [],
   "dockerImageVersionId": 28755,
   "isGpuEnabled": false,
   "isInternetEnabled": false,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 8.154516,
   "end_time": "2026-07-03T14:44:28.972225+00:00",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-07-03T14:44:20.817709+00:00",
   "version": "2.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
