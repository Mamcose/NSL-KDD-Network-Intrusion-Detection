{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "BilgAg 2.ipynb",
      "version": "0.3.2",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "TPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "Qx_dyZdlSXMK",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import sys\n",
        "import sklearn\n",
        "import io\n",
        "import random"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "THyNO1v5Mzrx",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "train_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Train.csv'\n",
        "test_url = 'https://raw.githubusercontent.com/merteroglu/NSL-KDD-Network-Instrusion-Detection/master/NSL_KDD_Test.csv'\n",
        "\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Utaaqng3NOY0",
        "colab_type": "code",
        "outputId": "c479e10d-5553-4b8c-8ddf-d9dc0fd3a2a7",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        }
      },
      "source": [
        "col_names = [\"duration\",\"protocol_type\",\"service\",\"flag\",\"src_bytes\",\n",
        "    \"dst_bytes\",\"land\",\"wrong_fragment\",\"urgent\",\"hot\",\"num_failed_logins\",\n",
        "    \"logged_in\",\"num_compromised\",\"root_shell\",\"su_attempted\",\"num_root\",\n",
        "    \"num_file_creations\",\"num_shells\",\"num_access_files\",\"num_outbound_cmds\",\n",
        "    \"is_host_login\",\"is_guest_login\",\"count\",\"srv_count\",\"serror_rate\",\n",
        "    \"srv_serror_rate\",\"rerror_rate\",\"srv_rerror_rate\",\"same_srv_rate\",\n",
        "    \"diff_srv_rate\",\"srv_diff_host_rate\",\"dst_host_count\",\"dst_host_srv_count\",\n",
        "    \"dst_host_same_srv_rate\",\"dst_host_diff_srv_rate\",\"dst_host_same_src_port_rate\",\n",
        "    \"dst_host_srv_diff_host_rate\",\"dst_host_serror_rate\",\"dst_host_srv_serror_rate\",\n",
        "    \"dst_host_rerror_rate\",\"dst_host_srv_rerror_rate\",\"label\"]\n",
        "\n",
        "\n",
        "df = pd.read_csv(train_url,header=None, names = col_names)\n",
        "\n",
        "df_test = pd.read_csv(test_url, header=None, names = col_names)\n",
        "\n",
        "print('Dimensions of the Training set:',df.shape)\n",
        "print('Dimensions of the Test set:',df_test.shape)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Dimensions of the Training set: (125973, 42)\n",
            "Dimensions of the Test set: (22544, 42)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "_zKWRUDkWpiH",
        "colab_type": "code",
        "outputId": "eed2117e-e451-4426-950f-0841bdf040e6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 224
        }
      },
      "source": [
        "df.head(5)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>duration</th>\n",
              "      <th>protocol_type</th>\n",
              "      <th>service</th>\n",
              "      <th>flag</th>\n",
              "      <th>src_bytes</th>\n",
              "      <th>dst_bytes</th>\n",
              "      <th>land</th>\n",
              "      <th>wrong_fragment</th>\n",
              "      <th>urgent</th>\n",
              "      <th>hot</th>\n",
              "      <th>num_failed_logins</th>\n",
              "      <th>logged_in</th>\n",
              "      <th>num_compromised</th>\n",
              "      <th>root_shell</th>\n",
              "      <th>su_attempted</th>\n",
              "      <th>num_root</th>\n",
              "      <th>num_file_creations</th>\n",
              "      <th>num_shells</th>\n",
              "      <th>num_access_files</th>\n",
              "      <th>num_outbound_cmds</th>\n",
              "      <th>is_host_login</th>\n",
              "      <th>is_guest_login</th>\n",
              "      <th>count</th>\n",
              "      <th>srv_count</th>\n",
              "      <th>serror_rate</th>\n",
              "      <th>srv_serror_rate</th>\n",
              "      <th>rerror_rate</th>\n",
              "      <th>srv_rerror_rate</th>\n",
              "      <th>same_srv_rate</th>\n",
              "      <th>diff_srv_rate</th>\n",
              "      <th>srv_diff_host_rate</th>\n",
              "      <th>dst_host_count</th>\n",
              "      <th>dst_host_srv_count</th>\n",
              "      <th>dst_host_same_srv_rate</th>\n",
              "      <th>dst_host_diff_srv_rate</th>\n",
              "      <th>dst_host_same_src_port_rate</th>\n",
              "      <th>dst_host_srv_diff_host_rate</th>\n",
              "      <th>dst_host_serror_rate</th>\n",
              "      <th>dst_host_srv_serror_rate</th>\n",
              "      <th>dst_host_rerror_rate</th>\n",
              "      <th>dst_host_srv_rerror_rate</th>\n",
              "      <th>label</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>tcp</td>\n",
              "      <td>ftp_data</td>\n",
              "      <td>SF</td>\n",
              "      <td>491</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>2</td>\n",
              "      <td>2</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>150</td>\n",
              "      <td>25</td>\n",
              "      <td>0.17</td>\n",
              "      <td>0.03</td>\n",
              "      <td>0.17</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.05</td>\n",
              "      <td>0.00</td>\n",
              "      <td>normal</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0</td>\n",
              "      <td>udp</td>\n",
              "      <td>other</td>\n",
              "      <td>SF</td>\n",
              "      <td>146</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>13</td>\n",
              "      <td>1</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.08</td>\n",
              "      <td>0.15</td>\n",
              "      <td>0.00</td>\n",
              "      <td>255</td>\n",
              "      <td>1</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.60</td>\n",
              "      <td>0.88</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>normal</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0</td>\n",
              "      <td>tcp</td>\n",
              "      <td>private</td>\n",
              "      <td>S0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>123</td>\n",
              "      <td>6</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.05</td>\n",
              "      <td>0.07</td>\n",
              "      <td>0.00</td>\n",
              "      <td>255</td>\n",
              "      <td>26</td>\n",
              "      <td>0.10</td>\n",
              "      <td>0.05</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>1.00</td>\n",
              "      <td>1.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>neptune</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0</td>\n",
              "      <td>tcp</td>\n",
              "      <td>http</td>\n",
              "      <td>SF</td>\n",
              "      <td>232</td>\n",
              "      <td>8153</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>5</td>\n",
              "      <td>5</td>\n",
              "      <td>0.2</td>\n",
              "      <td>0.2</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>30</td>\n",
              "      <td>255</td>\n",
              "      <td>1.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.03</td>\n",
              "      <td>0.04</td>\n",
              "      <td>0.03</td>\n",
              "      <td>0.01</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.01</td>\n",
              "      <td>normal</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0</td>\n",
              "      <td>tcp</td>\n",
              "      <td>http</td>\n",
              "      <td>SF</td>\n",
              "      <td>199</td>\n",
              "      <td>420</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>1</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>30</td>\n",
              "      <td>32</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.09</td>\n",
              "      <td>255</td>\n",
              "      <td>255</td>\n",
              "      <td>1.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>0.00</td>\n",
              "      <td>normal</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "   duration protocol_type  ... dst_host_srv_rerror_rate    label\n",
              "0         0           tcp  ...                     0.00   normal\n",
              "1         0           udp  ...                     0.00   normal\n",
              "2         0           tcp  ...                     0.00  neptune\n",
              "3         0           tcp  ...                     0.01   normal\n",
              "4         0           tcp  ...                     0.00   normal\n",
              "\n",
              "[5 rows x 42 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BAa8_nBFWwxf",
        "colab_type": "code",
        "outputId": "380fc920-3704-4671-f9f0-40ff93fe6a73",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1139
        }
      },
      "source": [
        "print('Label distribution Training set:')\n",
        "print(df['label'].value_counts())\n",
        "print()\n",
        "print('Label distribution Test set:')\n",
        "print(df_test['label'].value_counts())"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Label distribution Training set:\n",
            "normal             67343\n",
            "neptune            41214\n",
            "satan               3633\n",
            "ipsweep             3599\n",
            "portsweep           2931\n",
            "smurf               2646\n",
            "nmap                1493\n",
            "back                 956\n",
            "teardrop             892\n",
            "warezclient          890\n",
            "pod                  201\n",
            "guess_passwd          53\n",
            "buffer_overflow       30\n",
            "warezmaster           20\n",
            "land                  18\n",
            "imap                  11\n",
            "rootkit               10\n",
            "loadmodule             9\n",
            "ftp_write              8\n",
            "multihop               7\n",
            "phf                    4\n",
            "perl                   3\n",
            "spy                    2\n",
            "Name: label, dtype: int64\n",
            "\n",
            "Label distribution Test set:\n",
            "normal             9711\n",
            "neptune            4657\n",
            "guess_passwd       1231\n",
            "mscan               996\n",
            "warezmaster         944\n",
            "apache2             737\n",
            "satan               735\n",
            "processtable        685\n",
            "smurf               665\n",
            "back                359\n",
            "snmpguess           331\n",
            "saint               319\n",
            "mailbomb            293\n",
            "snmpgetattack       178\n",
            "portsweep           157\n",
            "ipsweep             141\n",
            "httptunnel          133\n",
            "nmap                 73\n",
            "pod                  41\n",
            "buffer_overflow      20\n",
            "multihop             18\n",
            "named                17\n",
            "ps                   15\n",
            "sendmail             14\n",
            "rootkit              13\n",
            "xterm                13\n",
            "teardrop             12\n",
            "xlock                 9\n",
            "land                  7\n",
            "xsnoop                4\n",
            "ftp_write             3\n",
            "loadmodule            2\n",
            "sqlattack             2\n",
            "worm                  2\n",
            "perl                  2\n",
            "phf                   2\n",
            "udpstorm              2\n",
            "imap                  1\n",
            "Name: label, dtype: int64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vFUPOfTWW7jF",
        "colab_type": "text"
      },
      "source": [
        "**Step 1: Data preprocessing:**\n",
        "\n",
        "One-Hot-Encoding, tüm kategorik özellikleri ikili özelliklere dönüştürmek için kullanılır. One-Hot-Endcoding gereksinimi, bu transformatöre giriş, kategorik(ayrık) özelliklerle alınan değerleri ifade eden bir tam sayı matrisi olmalıdır. Çıktı, her bir sütunun olası bir değere karşılık geldiği seyrek bir matris olacaktır. Giriş özelliklerinin [0, n_values] aralığında değerler aldıkları varsayılmaktadır. Bu nedenle her kategoriyi bir sayıya dönüştürmek için özelliklerin öncelikle LabelEncoder ile dönüştürülmesi gerekir."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TyLnY-XCXVnk",
        "colab_type": "code",
        "outputId": "7c1ed9f4-584c-4039-e4d4-429649c5c3dc",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        }
      },
      "source": [
        "# sütunlar kategorik, henüz binary değil: protocol_type (column 2), service (column 3), flag (column 4).\n",
        "\n",
        "print('Training set:')\n",
        "for col_name in df.columns:\n",
        "    if df[col_name].dtypes == 'object' :\n",
        "        unique_cat = len(df[col_name].unique())\n",
        "        print(\"Feature '{col_name}' has {unique_cat} categories\".format(col_name=col_name, unique_cat=unique_cat))\n",
        "\n",
        "print()\n",
        "print('Distribution of categories in service:')\n",
        "print(df['service'].value_counts().sort_values(ascending=False).head())"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Training set:\n",
            "Feature 'protocol_type' has 3 categories\n",
            "Feature 'service' has 70 categories\n",
            "Feature 'flag' has 11 categories\n",
            "Feature 'label' has 23 categories\n",
            "\n",
            "Distribution of categories in service:\n",
            "http        40338\n",
            "private     21853\n",
            "domain_u     9043\n",
            "smtp         7313\n",
            "ftp_data     6860\n",
            "Name: service, dtype: int64\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WhKXNXlwXb5G",
        "colab_type": "code",
        "outputId": "7d75d36f-24ba-426b-b6e1-fa0470eed0e1",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 102
        }
      },
      "source": [
        "# Test set\n",
        "print('Test set:')\n",
        "for col_name in df_test.columns:\n",
        "    if df_test[col_name].dtypes == 'object' :\n",
        "        unique_cat = len(df_test[col_name].unique())\n",
        "        print(\"Feature '{col_name}' has {unique_cat} categories\".format(col_name=col_name, unique_cat=unique_cat))\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Test set:\n",
            "Feature 'protocol_type' has 3 categories\n",
            "Feature 'service' has 64 categories\n",
            "Feature 'flag' has 11 categories\n",
            "Feature 'label' has 38 categories\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PZKviPW0XnzU",
        "colab_type": "text"
      },
      "source": [
        "**LabelEncoder**\n",
        "\n",
        "**Insert categorical features into a 2D numpy array**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SYmvn2qWXxs2",
        "colab_type": "code",
        "outputId": "c2351e4b-76b7-43c9-8a5a-226d0d95f6e9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        }
      },
      "source": [
        "from sklearn.preprocessing import LabelEncoder,OneHotEncoder\n",
        "categorical_columns=['protocol_type', 'service', 'flag']\n",
        "\n",
        "df_categorical_values = df[categorical_columns]\n",
        "testdf_categorical_values = df_test[categorical_columns]\n",
        "\n",
        "df_categorical_values.head()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>protocol_type</th>\n",
              "      <th>service</th>\n",
              "      <th>flag</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>tcp</td>\n",
              "      <td>ftp_data</td>\n",
              "      <td>SF</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>udp</td>\n",
              "      <td>other</td>\n",
              "      <td>SF</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>tcp</td>\n",
              "      <td>private</td>\n",
              "      <td>S0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>tcp</td>\n",
              "      <td>http</td>\n",
              "      <td>SF</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>tcp</td>\n",
              "      <td>http</td>\n",
              "      <td>SF</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  protocol_type   service flag\n",
              "0           tcp  ftp_data   SF\n",
              "1           udp     other   SF\n",
              "2           tcp   private   S0\n",
              "3           tcp      http   SF\n",
              "4           tcp      http   SF"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DNVa7mN9X1qV",
        "colab_type": "code",
        "outputId": "98b0fcda-8a81-4ec4-bf77-df8c24d9c420",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 88
        }
      },
      "source": [
        "# protocol type\n",
        "unique_protocol=sorted(df.protocol_type.unique())\n",
        "string1 = 'Protocol_type_'\n",
        "unique_protocol2=[string1 + x for x in unique_protocol]\n",
        "print(unique_protocol2)\n",
        "\n",
        "# service\n",
        "unique_service=sorted(df.service.unique())\n",
        "string2 = 'service_'\n",
        "unique_service2=[string2 + x for x in unique_service]\n",
        "print(unique_service2)\n",
        "\n",
        "\n",
        "# flag\n",
        "unique_flag=sorted(df.flag.unique())\n",
        "string3 = 'flag_'\n",
        "unique_flag2=[string3 + x for x in unique_flag]\n",
        "print(unique_flag2)\n",
        "\n",
        "\n",
        "# put together\n",
        "dumcols=unique_protocol2 + unique_service2 + unique_flag2\n",
        "\n",
        "\n",
        "#do it for test set\n",
        "unique_service_test=sorted(df_test.service.unique())\n",
        "unique_service2_test=[string2 + x for x in unique_service_test]\n",
        "testdumcols=unique_protocol2 + unique_service2_test + unique_flag2\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "['Protocol_type_icmp', 'Protocol_type_tcp', 'Protocol_type_udp']\n",
            "['service_IRC', 'service_X11', 'service_Z39_50', 'service_aol', 'service_auth', 'service_bgp', 'service_courier', 'service_csnet_ns', 'service_ctf', 'service_daytime', 'service_discard', 'service_domain', 'service_domain_u', 'service_echo', 'service_eco_i', 'service_ecr_i', 'service_efs', 'service_exec', 'service_finger', 'service_ftp', 'service_ftp_data', 'service_gopher', 'service_harvest', 'service_hostnames', 'service_http', 'service_http_2784', 'service_http_443', 'service_http_8001', 'service_imap4', 'service_iso_tsap', 'service_klogin', 'service_kshell', 'service_ldap', 'service_link', 'service_login', 'service_mtp', 'service_name', 'service_netbios_dgm', 'service_netbios_ns', 'service_netbios_ssn', 'service_netstat', 'service_nnsp', 'service_nntp', 'service_ntp_u', 'service_other', 'service_pm_dump', 'service_pop_2', 'service_pop_3', 'service_printer', 'service_private', 'service_red_i', 'service_remote_job', 'service_rje', 'service_shell', 'service_smtp', 'service_sql_net', 'service_ssh', 'service_sunrpc', 'service_supdup', 'service_systat', 'service_telnet', 'service_tftp_u', 'service_tim_i', 'service_time', 'service_urh_i', 'service_urp_i', 'service_uucp', 'service_uucp_path', 'service_vmnet', 'service_whois']\n",
            "['flag_OTH', 'flag_REJ', 'flag_RSTO', 'flag_RSTOS0', 'flag_RSTR', 'flag_S0', 'flag_S1', 'flag_S2', 'flag_S3', 'flag_SF', 'flag_SH']\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Nf-QhZyfX9Jd",
        "colab_type": "text"
      },
      "source": [
        "**Transform categorical features into numbers using LabelEncoder()**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ta0CsfRUX5Tg",
        "colab_type": "code",
        "outputId": "b678107b-e3be-4cb5-a4b0-4494ea5e072b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 238
        }
      },
      "source": [
        "df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)\n",
        "\n",
        "print(df_categorical_values.head())\n",
        "print('--------------------')\n",
        "print(df_categorical_values_enc.head())\n",
        "\n",
        "# test set\n",
        "testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "  protocol_type   service flag\n",
            "0           tcp  ftp_data   SF\n",
            "1           udp     other   SF\n",
            "2           tcp   private   S0\n",
            "3           tcp      http   SF\n",
            "4           tcp      http   SF\n",
            "--------------------\n",
            "   protocol_type  service  flag\n",
            "0              1       20     9\n",
            "1              2       44     9\n",
            "2              1       49     5\n",
            "3              1       24     9\n",
            "4              1       24     9\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "iq_pnkScYDOi",
        "colab_type": "text"
      },
      "source": [
        "**One-Hot-Encoding**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MVBytj5dYGDu",
        "colab_type": "code",
        "outputId": "f2c9a777-ea4e-4181-fcd6-099c9152cc8f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 253
        }
      },
      "source": [
        "enc = OneHotEncoder(categories='auto')\n",
        "df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)\n",
        "df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)\n",
        "\n",
        "\n",
        "# test set\n",
        "testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)\n",
        "testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)\n",
        "\n",
        "df_cat_data.head()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Protocol_type_icmp</th>\n",
              "      <th>Protocol_type_tcp</th>\n",
              "      <th>Protocol_type_udp</th>\n",
              "      <th>service_IRC</th>\n",
              "      <th>service_X11</th>\n",
              "      <th>service_Z39_50</th>\n",
              "      <th>service_aol</th>\n",
              "      <th>service_auth</th>\n",
              "      <th>service_bgp</th>\n",
              "      <th>service_courier</th>\n",
              "      <th>service_csnet_ns</th>\n",
              "      <th>service_ctf</th>\n",
              "      <th>service_daytime</th>\n",
              "      <th>service_discard</th>\n",
              "      <th>service_domain</th>\n",
              "      <th>service_domain_u</th>\n",
              "      <th>service_echo</th>\n",
              "      <th>service_eco_i</th>\n",
              "      <th>service_ecr_i</th>\n",
              "      <th>service_efs</th>\n",
              "      <th>service_exec</th>\n",
              "      <th>service_finger</th>\n",
              "      <th>service_ftp</th>\n",
              "      <th>service_ftp_data</th>\n",
              "      <th>service_gopher</th>\n",
              "      <th>service_harvest</th>\n",
              "      <th>service_hostnames</th>\n",
              "      <th>service_http</th>\n",
              "      <th>service_http_2784</th>\n",
              "      <th>service_http_443</th>\n",
              "      <th>service_http_8001</th>\n",
              "      <th>service_imap4</th>\n",
              "      <th>service_iso_tsap</th>\n",
              "      <th>service_klogin</th>\n",
              "      <th>service_kshell</th>\n",
              "      <th>service_ldap</th>\n",
              "      <th>service_link</th>\n",
              "      <th>service_login</th>\n",
              "      <th>service_mtp</th>\n",
              "      <th>service_name</th>\n",
              "      <th>...</th>\n",
              "      <th>service_nnsp</th>\n",
              "      <th>service_nntp</th>\n",
              "      <th>service_ntp_u</th>\n",
              "      <th>service_other</th>\n",
              "      <th>service_pm_dump</th>\n",
              "      <th>service_pop_2</th>\n",
              "      <th>service_pop_3</th>\n",
              "      <th>service_printer</th>\n",
              "      <th>service_private</th>\n",
              "      <th>service_red_i</th>\n",
              "      <th>service_remote_job</th>\n",
              "      <th>service_rje</th>\n",
              "      <th>service_shell</th>\n",
              "      <th>service_smtp</th>\n",
              "      <th>service_sql_net</th>\n",
              "      <th>service_ssh</th>\n",
              "      <th>service_sunrpc</th>\n",
              "      <th>service_supdup</th>\n",
              "      <th>service_systat</th>\n",
              "      <th>service_telnet</th>\n",
              "      <th>service_tftp_u</th>\n",
              "      <th>service_tim_i</th>\n",
              "      <th>service_time</th>\n",
              "      <th>service_urh_i</th>\n",
              "      <th>service_urp_i</th>\n",
              "      <th>service_uucp</th>\n",
              "      <th>service_uucp_path</th>\n",
              "      <th>service_vmnet</th>\n",
              "      <th>service_whois</th>\n",
              "      <th>flag_OTH</th>\n",
              "      <th>flag_REJ</th>\n",
              "      <th>flag_RSTO</th>\n",
              "      <th>flag_RSTOS0</th>\n",
              "      <th>flag_RSTR</th>\n",
              "      <th>flag_S0</th>\n",
              "      <th>flag_S1</th>\n",
              "      <th>flag_S2</th>\n",
              "      <th>flag_S3</th>\n",
              "      <th>flag_SF</th>\n",
              "      <th>flag_SH</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>0.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 84 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "   Protocol_type_icmp  Protocol_type_tcp  ...  flag_SF  flag_SH\n",
              "0                 0.0                1.0  ...      1.0      0.0\n",
              "1                 0.0                0.0  ...      1.0      0.0\n",
              "2                 0.0                1.0  ...      0.0      0.0\n",
              "3                 0.0                1.0  ...      1.0      0.0\n",
              "4                 0.0                1.0  ...      1.0      0.0\n",
              "\n",
              "[5 rows x 84 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7D-ZvKY6bS-r",
        "colab_type": "text"
      },
      "source": [
        "**Test setteki eksik sütunlar eklenir**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HxTdPirkbW-V",
        "colab_type": "code",
        "outputId": "fb55b332-5cfb-4bb6-e1f1-11fceb0029fd",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 119
        }
      },
      "source": [
        "trainservice=df['service'].tolist()\n",
        "testservice= df_test['service'].tolist()\n",
        "difference=list(set(trainservice) - set(testservice))\n",
        "string = 'service_'\n",
        "difference=[string + x for x in difference]\n",
        "difference"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "['service_http_2784',\n",
              " 'service_http_8001',\n",
              " 'service_red_i',\n",
              " 'service_aol',\n",
              " 'service_harvest',\n",
              " 'service_urh_i']"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tpBVYECtcN02",
        "colab_type": "code",
        "outputId": "f39dc7be-61a3-4987-cbae-db11747227ab",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        }
      },
      "source": [
        "for col in difference:\n",
        "    testdf_cat_data[col] = 0\n",
        "\n",
        "print(df_cat_data.shape)    \n",
        "print(testdf_cat_data.shape)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(125973, 84)\n",
            "(22544, 84)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "4Vsl_YcrcSd8",
        "colab_type": "text"
      },
      "source": [
        "**Ana dataframe'e yeni sayısal sütunlar eklenir**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "KOXgIdAZcVAF",
        "colab_type": "code",
        "outputId": "daa03d97-1f70-435d-91d7-18af484becbf",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 51
        }
      },
      "source": [
        "newdf=df.join(df_cat_data)\n",
        "newdf.drop('flag', axis=1, inplace=True)\n",
        "newdf.drop('protocol_type', axis=1, inplace=True)\n",
        "newdf.drop('service', axis=1, inplace=True)\n",
        "\n",
        "# test data\n",
        "newdf_test=df_test.join(testdf_cat_data)\n",
        "newdf_test.drop('flag', axis=1, inplace=True)\n",
        "newdf_test.drop('protocol_type', axis=1, inplace=True)\n",
        "newdf_test.drop('service', axis=1, inplace=True)\n",
        "\n",
        "print(newdf.shape)\n",
        "print(newdf_test.shape)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(125973, 123)\n",
            "(22544, 123)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dZsTN_Xoca_b",
        "colab_type": "text"
      },
      "source": [
        " Dataset her atak kategorisi için ayrı datasetlere ayrıldı. Atak etiketleri her biri için yeniden adlandırıldı. 0=Normal, 1=DoS, 2=Probe, 3=R2L, 4=U2R. Yeni datasetlerde etiket sütunu yeni değerler ile değiştirildi.\n",
        " \n",
        " DoS : \n",
        " \n",
        " Probe : \n",
        " \n",
        " R2L :\n",
        " \n",
        " U2R :"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "706-Qqhycc_-",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "labeldf=newdf['label']\n",
        "labeldf_test=newdf_test['label']\n",
        "\n",
        "\n",
        "# change the label column\n",
        "newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,\n",
        "                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2\n",
        "                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,\n",
        "                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})\n",
        "newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,\n",
        "                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2\n",
        "                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,\n",
        "                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})\n",
        "\n",
        "\n",
        "\n",
        "# put the new label column back\n",
        "newdf['label'] = newlabeldf\n",
        "newdf_test['label'] = newlabeldf_test"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Bu7S66qJcw56",
        "colab_type": "code",
        "outputId": "ca5533bc-1d89-4169-b777-de8db54a1140",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 204
        }
      },
      "source": [
        "to_drop_DoS = [0,1]\n",
        "to_drop_Probe = [0,2]\n",
        "to_drop_R2L = [0,3]\n",
        "to_drop_U2R = [0,4]\n",
        "\n",
        "# Kendisi dışındaki label değerine sahip tüm satırları filtrele\n",
        "# isin filter function\n",
        "\n",
        "DoS_df=newdf[newdf['label'].isin(to_drop_DoS)];\n",
        "Probe_df=newdf[newdf['label'].isin(to_drop_Probe)];\n",
        "R2L_df=newdf[newdf['label'].isin(to_drop_R2L)];\n",
        "U2R_df=newdf[newdf['label'].isin(to_drop_U2R)];\n",
        "\n",
        "\n",
        "\n",
        "#test\n",
        "DoS_df_test=newdf_test[newdf_test['label'].isin(to_drop_DoS)];\n",
        "Probe_df_test=newdf_test[newdf_test['label'].isin(to_drop_Probe)];\n",
        "R2L_df_test=newdf_test[newdf_test['label'].isin(to_drop_R2L)];\n",
        "U2R_df_test=newdf_test[newdf_test['label'].isin(to_drop_U2R)];\n",
        "\n",
        "\n",
        "print('Train:')\n",
        "print('Dimensions of DoS:' ,DoS_df.shape)\n",
        "print('Dimensions of Probe:' ,Probe_df.shape)\n",
        "print('Dimensions of R2L:' ,R2L_df.shape)\n",
        "print('Dimensions of U2R:' ,U2R_df.shape)\n",
        "print()\n",
        "print('Test:')\n",
        "print('Dimensions of DoS:' ,DoS_df_test.shape)\n",
        "print('Dimensions of Probe:' ,Probe_df_test.shape)\n",
        "print('Dimensions of R2L:' ,R2L_df_test.shape)\n",
        "print('Dimensions of U2R:' ,U2R_df_test.shape)"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train:\n",
            "Dimensions of DoS: (113270, 123)\n",
            "Dimensions of Probe: (78999, 123)\n",
            "Dimensions of R2L: (68338, 123)\n",
            "Dimensions of U2R: (67395, 123)\n",
            "\n",
            "Test:\n",
            "Dimensions of DoS: (17171, 123)\n",
            "Dimensions of Probe: (12132, 123)\n",
            "Dimensions of R2L: (12596, 123)\n",
            "Dimensions of U2R: (9778, 123)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Dvi-Oglzc5Bs",
        "colab_type": "text"
      },
      "source": [
        "**Step 2: Feature Scaling**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-VeQhD09c6qT",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "# Split dataframes into X & Y\n",
        "# X Özellikler , Y sonuç değişkenleri\n",
        "\n",
        "X_DoS = DoS_df.drop('label',1)\n",
        "Y_DoS = DoS_df.label\n",
        "\n",
        "X_Probe = Probe_df.drop('label',1)\n",
        "Y_Probe = Probe_df.label\n",
        "\n",
        "X_R2L = R2L_df.drop('label',1)\n",
        "Y_R2L = R2L_df.label\n",
        "\n",
        "X_U2R = U2R_df.drop('label',1)\n",
        "Y_U2R = U2R_df.label\n",
        "\n",
        "# test set\n",
        "X_DoS_test = DoS_df_test.drop('label',1)\n",
        "Y_DoS_test = DoS_df_test.label\n",
        "\n",
        "X_Probe_test = Probe_df_test.drop('label',1)\n",
        "Y_Probe_test = Probe_df_test.label\n",
        "\n",
        "X_R2L_test = R2L_df_test.drop('label',1)\n",
        "Y_R2L_test = R2L_df_test.label\n",
        "\n",
        "X_U2R_test = U2R_df_test.drop('label',1)\n",
        "Y_U2R_test = U2R_df_test.label\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "yl7gee3KdAb5",
        "colab_type": "text"
      },
      "source": [
        "**Sütun isimleri bu aşamada silineceği için daha sonra kullanmak üzere sütun isimlerini kayıt ederiz.**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2nWS5oukdFS6",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "colNames=list(X_DoS)\n",
        "colNames_test=list(X_DoS_test)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FEBafDjodRcT",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn import preprocessing\n",
        "\n",
        "scaler1 = preprocessing.StandardScaler().fit(X_DoS)\n",
        "X_DoS=scaler1.transform(X_DoS) \n",
        "\n",
        "scaler2 = preprocessing.StandardScaler().fit(X_Probe)\n",
        "X_Probe=scaler2.transform(X_Probe)\n",
        "\n",
        "scaler3 = preprocessing.StandardScaler().fit(X_R2L)\n",
        "X_R2L=scaler3.transform(X_R2L)\n",
        "\n",
        "scaler4 = preprocessing.StandardScaler().fit(X_U2R)\n",
        "X_U2R=scaler4.transform(X_U2R) \n",
        "\n",
        "# test data\n",
        "scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)\n",
        "X_DoS_test=scaler5.transform(X_DoS_test) \n",
        "\n",
        "scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)\n",
        "X_Probe_test=scaler6.transform(X_Probe_test) \n",
        "\n",
        "scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)\n",
        "X_R2L_test=scaler7.transform(X_R2L_test) \n",
        "\n",
        "scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)\n",
        "X_U2R_test=scaler8.transform(X_U2R_test)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pc2v2hb9dc8s",
        "colab_type": "text"
      },
      "source": [
        "**Step 3: Feature Selection:**\n",
        "\n",
        "---"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "k04dKGOWepgB",
        "colab_type": "text"
      },
      "source": [
        "**Recursive Feature Elimination (RFE) , en iyi 13 özellik (grup olarak)**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oR4xlXjlM2_M",
        "colab_type": "text"
      },
      "source": [
        "# Random Forest"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "paIjq2cci39w",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from sklearn.feature_selection import RFE\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "\n",
        "clf = RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "rfe = RFE(estimator=clf, n_features_to_select=13, step=1)\n",
        "\n",
        "rfe.fit(X_DoS, Y_DoS.astype(int))\n",
        "X_rfeDoS=rfe.transform(X_DoS)\n",
        "true=rfe.support_\n",
        "rfecolindex_DoS=[i for i, x in enumerate(true) if x]\n",
        "rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WLTTTvNIjcSc",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "rfe.fit(X_Probe, Y_Probe.astype(int))\n",
        "X_rfeProbe=rfe.transform(X_Probe)\n",
        "true=rfe.support_\n",
        "rfecolindex_Probe=[i for i, x in enumerate(true) if x]\n",
        "rfecolname_Probe=list(colNames[i] for i in rfecolindex_Probe)\n"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EdT_PCLPkHiu",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "rfe.fit(X_R2L, Y_R2L.astype(int))\n",
        "X_rfeR2L=rfe.transform(X_R2L)\n",
        "true=rfe.support_\n",
        "rfecolindex_R2L=[i for i, x in enumerate(true) if x]\n",
        "rfecolname_R2L=list(colNames[i] for i in rfecolindex_R2L)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "G9DtjxmDkhTL",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "rfe.fit(X_U2R, Y_U2R.astype(int))\n",
        "X_rfeU2R=rfe.transform(X_U2R)\n",
        "true=rfe.support_\n",
        "rfecolindex_U2R=[i for i, x in enumerate(true) if x]\n",
        "rfecolname_U2R=list(colNames[i] for i in rfecolindex_U2R)"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "z07YGhS3k8tN",
        "colab_type": "text"
      },
      "source": [
        "**Summary of features selected by RFE**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ej29Qdlxk6KO",
        "colab_type": "code",
        "outputId": "ec6bacc6-457c-4b40-daeb-a103f8101ffb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 156
        }
      },
      "source": [
        "print('Features selected for DoS:',rfecolname_DoS)\n",
        "print()\n",
        "print('Features selected for Probe:',rfecolname_Probe)\n",
        "print()\n",
        "print('Features selected for R2L:',rfecolname_R2L)\n",
        "print()\n",
        "print('Features selected for U2R:',rfecolname_U2R)\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Features selected for DoS: ['src_bytes', 'dst_bytes', 'wrong_fragment', 'count', 'srv_count', 'same_srv_rate', 'diff_srv_rate', 'dst_host_same_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_serror_rate', 'Protocol_type_icmp', 'flag_S0', 'flag_SF']\n",
            "\n",
            "Features selected for Probe: ['src_bytes', 'dst_bytes', 'count', 'rerror_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_rerror_rate', 'service_eco_i', 'service_private']\n",
            "\n",
            "Features selected for R2L: ['duration', 'src_bytes', 'dst_bytes', 'hot', 'is_guest_login', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'service_ftp', 'service_ftp_data']\n",
            "\n",
            "Features selected for U2R: ['duration', 'src_bytes', 'dst_bytes', 'hot', 'num_compromised', 'root_shell', 'num_file_creations', 'count', 'dst_host_count', 'dst_host_srv_count', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'service_ftp_data']\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "25gCTl0HlCrP",
        "colab_type": "code",
        "outputId": "1b7dde02-93a8-4155-c8ad-a5409b17f9f0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "print(X_rfeDoS.shape)\n",
        "print(X_rfeProbe.shape)\n",
        "print(X_rfeR2L.shape)\n",
        "print(X_rfeU2R.shape)\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(113270, 13)\n",
            "(78999, 13)\n",
            "(68338, 13)\n",
            "(67395, 13)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UIy12t5AlHMB",
        "colab_type": "text"
      },
      "source": [
        "**Step 4: Build the model:**\n",
        "\n",
        "Classifier is trained for all features and for reduced features, for later comparison.\n",
        "\n",
        "The classifier model itself is stored in the clf variable."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uw1m5wUelNr8",
        "colab_type": "code",
        "outputId": "d9cc05e2-1269-4501-f606-3a15b9f53b6a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 136
        }
      },
      "source": [
        "# all features\n",
        "clf_DoS=RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "clf_Probe=RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "clf_R2L=RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "clf_U2R=RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "clf_DoS.fit(X_DoS, Y_DoS.astype(int))\n",
        "clf_Probe.fit(X_Probe, Y_Probe.astype(int))\n",
        "clf_R2L.fit(X_R2L, Y_R2L.astype(int))\n",
        "clf_U2R.fit(X_U2R, Y_U2R.astype(int))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n",
              "                       max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
              "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
              "                       min_samples_leaf=1, min_samples_split=2,\n",
              "                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=2,\n",
              "                       oob_score=False, random_state=None, verbose=0,\n",
              "                       warm_start=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hBAfHdBdlnlW",
        "colab_type": "code",
        "outputId": "ddc9d26d-f33b-4492-e00f-85c712a807cf",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 136
        }
      },
      "source": [
        "# selected features\n",
        "clf_rfeDoS=RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "clf_rfeProbe=RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "clf_rfeR2L=RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "clf_rfeU2R=RandomForestClassifier(n_estimators=10,n_jobs=2)\n",
        "clf_rfeDoS.fit(X_rfeDoS, Y_DoS.astype(int))\n",
        "clf_rfeProbe.fit(X_rfeProbe, Y_Probe.astype(int))\n",
        "clf_rfeR2L.fit(X_rfeR2L, Y_R2L.astype(int))\n",
        "clf_rfeU2R.fit(X_rfeU2R, Y_U2R.astype(int))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n",
              "                       max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
              "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
              "                       min_samples_leaf=1, min_samples_split=2,\n",
              "                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=2,\n",
              "                       oob_score=False, random_state=None, verbose=0,\n",
              "                       warm_start=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 28
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "1dnH9hPel4Pl",
        "colab_type": "text"
      },
      "source": [
        "**Step 5: Prediction & Evaluation (validation):**\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "\n",
        "Using all Features for each category\n",
        "\n",
        "Confusion Matrices\n",
        "\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "\n",
        "DoS¶"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hwR59uW4l_Iy",
        "colab_type": "code",
        "outputId": "6c887bd2-c98c-4bfe-cafb-86b62181cb45",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# Apply the classifier we trained to the test data (which it has never seen before)\n",
        "clf_DoS.predict(X_DoS_test)\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([1, 1, 0, ..., 0, 0, 0])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 29
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3ilISJ2rmGWW",
        "colab_type": "code",
        "outputId": "6f145b10-a3ed-4662-acb2-a183db2a8c46",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 187
        }
      },
      "source": [
        "# View the predicted probabilities of the first 10 observations\n",
        "clf_DoS.predict_proba(X_DoS_test)[0:10]"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.4, 0.6],\n",
              "       [0.3, 0.7],\n",
              "       [1. , 0. ],\n",
              "       [1. , 0. ],\n",
              "       [0.6, 0.4],\n",
              "       [0.8, 0.2],\n",
              "       [0.9, 0.1],\n",
              "       [0.3, 0.7],\n",
              "       [0.6, 0.4],\n",
              "       [1. , 0. ]])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 30
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5BEe8Ar1mH83",
        "colab_type": "code",
        "outputId": "0abaa0e2-3882-4979-b253-cee41fb0efbc",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_DoS_pred=clf_DoS.predict(X_DoS_test)\n",
        "\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9653</td>\n",
              "      <td>58</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>4316</td>\n",
              "      <td>3144</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     1\n",
              "Actual attacks               \n",
              "0                  9653    58\n",
              "1                  4316  3144"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 31
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "CHQRCzW9mPGx",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xUIiP1bZmdeR",
        "colab_type": "code",
        "outputId": "f36acf64-d3a4-4466-ef3b-bb9c62dc623f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_Probe_pred=clf_Probe.predict(X_Probe_test)\n",
        "# Create confusion matrix\n",
        "\n",
        "pd.crosstab(Y_Probe_test, Y_Probe_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>2</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9393</td>\n",
              "      <td>318</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>694</td>\n",
              "      <td>1727</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     2\n",
              "Actual attacks               \n",
              "0                  9393   318\n",
              "2                   694  1727"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 32
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vm92mbNfmjF4",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "z7ro_ovNmkIz",
        "colab_type": "code",
        "outputId": "5be82c6a-90dc-4984-e7b9-1d721b648f8a",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_R2L_pred=clf_R2L.predict(X_R2L_test)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_R2L_test, Y_R2L_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>3</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9711</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>2858</td>\n",
              "      <td>27</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0   3\n",
              "Actual attacks             \n",
              "0                  9711   0\n",
              "3                  2858  27"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 33
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2sXrTdVHmpjr",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "b8MtOKsemrAN",
        "colab_type": "code",
        "outputId": "23ce2a70-ee01-4d3c-b7ef-a42f58faed39",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_U2R_pred=clf_U2R.predict(X_U2R_test)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_U2R_test, Y_U2R_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9711</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>67</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0\n",
              "Actual attacks         \n",
              "0                  9711\n",
              "4                    67"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 34
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "NLCyve_1muNe",
        "colab_type": "text"
      },
      "source": [
        "**Cross Validation: Accuracy, Precision, Recall, F-measure**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "nEqJCtJUm2wz",
        "colab_type": "text"
      },
      "source": [
        "**DoS**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "smwc3tZUmvaf",
        "colab_type": "code",
        "outputId": "4d791e8e-7813-41af-fd94-69cf0e1c182b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn import metrics\n",
        "accuracy = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='precision')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='recall')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='f1')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99802 (+/- 0.00140)\n",
            "Precision: 0.99879 (+/- 0.00253)\n",
            "Recall: 0.99665 (+/- 0.00275)\n",
            "F-measure: 0.99785 (+/- 0.00215)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "isVUQMo6m5Gu",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "0UQ7ZKMZm6Ye",
        "colab_type": "code",
        "outputId": "55bbce90-5c07-40de-d33d-8ccb91e18f7c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99662 (+/- 0.00325)\n",
            "Precision: 0.99721 (+/- 0.00427)\n",
            "Recall: 0.99349 (+/- 0.00517)\n",
            "F-measure: 0.99378 (+/- 0.00345)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oDtTPPlgnPky",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TdxuBHTrnO7W",
        "colab_type": "code",
        "outputId": "98f7d852-84a2-4acc-f85a-ced58dbe5964",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99765 (+/- 0.00225)\n",
            "Precision: 0.96396 (+/- 0.11062)\n",
            "Recall: 0.83432 (+/- 0.15130)\n",
            "F-measure: 0.91707 (+/- 0.06314)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rRdxQ64LwtBj",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MrfDjRonwqSr",
        "colab_type": "code",
        "outputId": "163fde91-e9ce-4110-8674-abfd0155d8d0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.98079 (+/- 0.00611)\n",
            "Precision: 0.97390 (+/- 0.00864)\n",
            "Recall: 0.97019 (+/- 0.00967)\n",
            "F-measure: 0.97256 (+/- 0.00604)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pNJ5P6K8nX61",
        "colab_type": "text"
      },
      "source": [
        "**Using 13 Features for each category**\n",
        "\n",
        "\n",
        "Confusion Matrices\n",
        "\n",
        "DoS"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PI--9Z3xnZPL",
        "colab_type": "code",
        "outputId": "4058e22f-c152-4f0b-b881-5b27b4693c4e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 34
        }
      },
      "source": [
        "# reduce test dataset to 13 features, use only features described in rfecolname_DoS etc.\n",
        "X_DoS_test2=X_DoS_test[:,rfecolindex_DoS]\n",
        "X_Probe_test2=X_Probe_test[:,rfecolindex_Probe]\n",
        "X_R2L_test2=X_R2L_test[:,rfecolindex_R2L]\n",
        "X_U2R_test2=X_U2R_test[:,rfecolindex_U2R]\n",
        "X_U2R_test2.shape"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(9778, 13)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 39
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dvfVjn1IpdkF",
        "colab_type": "code",
        "outputId": "5d7232e5-3162-40cf-e8ba-c7051f010bd6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_DoS_pred2=clf_rfeDoS.predict(X_DoS_test2)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_DoS_test, Y_DoS_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9662</td>\n",
              "      <td>49</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>7398</td>\n",
              "      <td>62</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0   1\n",
              "Actual attacks             \n",
              "0                  9662  49\n",
              "1                  7398  62"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 40
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "5pNtpNWFpjqc",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bZk-GxAOpiDy",
        "colab_type": "code",
        "outputId": "b4c3fc2d-649f-4579-874c-3c3d951216cf",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_Probe_pred2=clf_rfeProbe.predict(X_Probe_test2)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_Probe_test, Y_Probe_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>2</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9444</td>\n",
              "      <td>267</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1217</td>\n",
              "      <td>1204</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     2\n",
              "Actual attacks               \n",
              "0                  9444   267\n",
              "2                  1217  1204"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 41
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dhP4467dpoon",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "itCdtf4Opykb",
        "colab_type": "code",
        "outputId": "fee21e54-d04d-47ca-ae56-aa3374ab691b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_R2L_pred2=clf_rfeR2L.predict(X_R2L_test2)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_R2L_test, Y_R2L_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9711</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>2885</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0\n",
              "Actual attacks         \n",
              "0                  9711\n",
              "3                  2885"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 42
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_CuyYSWMp3wP",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LB5Sve1Xp5Qz",
        "colab_type": "code",
        "outputId": "4f5b8012-3b8a-43db-91a3-89d33fb58709",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_U2R_pred2=clf_rfeU2R.predict(X_U2R_test2)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_U2R_test, Y_U2R_pred2, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>4</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9711</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>54</td>\n",
              "      <td>13</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0   4\n",
              "Actual attacks             \n",
              "0                  9711   0\n",
              "4                    54  13"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 43
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "7-Nz367hp8oG",
        "colab_type": "text"
      },
      "source": [
        "**Cross Validation: Accuracy, Precision, Recall, F-measure**\n",
        "\n",
        "**DoS**\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eOq4IH2CqHxi",
        "colab_type": "code",
        "outputId": "c026297d-5433-411e-9fdb-a4026bebe962",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='precision')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='recall')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_rfeDoS, X_DoS_test2, Y_DoS_test, cv=10, scoring='f1')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99767 (+/- 0.00346)\n",
            "Precision: 0.99812 (+/- 0.00452)\n",
            "Recall: 0.99584 (+/- 0.00593)\n",
            "F-measure: 0.99772 (+/- 0.00356)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6gkA-xV5qOJp",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ynF31fBiqPZV",
        "colab_type": "code",
        "outputId": "c732f20f-23fa-41ff-f680-0cf4009a1fa6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_rfeProbe, X_Probe_test2, Y_Probe_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99390 (+/- 0.00473)\n",
            "Precision: 0.99154 (+/- 0.00829)\n",
            "Recall: 0.98812 (+/- 0.01316)\n",
            "F-measure: 0.98992 (+/- 0.00795)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BWyw08Z8qVi5",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eng_RcAVqVHx",
        "colab_type": "code",
        "outputId": "2b782999-fa83-4f4a-e44b-5c549a63530b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_rfeR2L, X_R2L_test2, Y_R2L_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.97848 (+/- 0.00821)\n",
            "Precision: 0.97249 (+/- 0.01381)\n",
            "Recall: 0.96688 (+/- 0.01398)\n",
            "F-measure: 0.96877 (+/- 0.01130)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "omjGOf-9qdBY",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "zhkYhoP1qfAt",
        "colab_type": "code",
        "outputId": "b43e2565-d72e-49d2-aa67-b04772790e06",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_rfeU2R, X_U2R_test2, Y_U2R_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99744 (+/- 0.00307)\n",
            "Precision: 0.94527 (+/- 0.11860)\n",
            "Recall: 0.81294 (+/- 0.18438)\n",
            "F-measure: 0.87059 (+/- 0.15659)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "LMZrPfW_vStU",
        "colab_type": "text"
      },
      "source": [
        "# KNeighbors\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SxPf0tnRvU66",
        "colab_type": "code",
        "outputId": "6fc1d598-083a-4ff3-993b-d3849b70221e",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        }
      },
      "source": [
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "\n",
        "clf_KNN_DoS=KNeighborsClassifier()\n",
        "clf_KNN_Probe=KNeighborsClassifier()\n",
        "clf_KNN_R2L=KNeighborsClassifier()\n",
        "clf_KNN_U2R=KNeighborsClassifier()\n",
        "\n",
        "clf_KNN_DoS.fit(X_DoS, Y_DoS.astype(int))\n",
        "clf_KNN_Probe.fit(X_Probe, Y_Probe.astype(int))\n",
        "clf_KNN_R2L.fit(X_R2L, Y_R2L.astype(int))\n",
        "clf_KNN_U2R.fit(X_U2R, Y_U2R.astype(int))\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',\n",
              "                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,\n",
              "                     weights='uniform')"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 48
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OfLBLvwl6W0N",
        "colab_type": "text"
      },
      "source": [
        " **DoS**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "l2HYm4k_5zAt",
        "colab_type": "code",
        "outputId": "083df9b3-bfd8-4500-cb28-8f5e415ec884",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_DoS_pred=clf_KNN_DoS.predict(X_DoS_test)\n",
        "\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9422</td>\n",
              "      <td>289</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1573</td>\n",
              "      <td>5887</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     1\n",
              "Actual attacks               \n",
              "0                  9422   289\n",
              "1                  1573  5887"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 49
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "AloJ7mLN6fOy",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "m08qtIuM6Hc-",
        "colab_type": "code",
        "outputId": "e72804b0-6cfd-44ae-e955-558db9a347f6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_Probe_pred=clf_KNN_Probe.predict(X_Probe_test)\n",
        "# Create confusion matrix\n",
        "\n",
        "pd.crosstab(Y_Probe_test, Y_Probe_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>2</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9437</td>\n",
              "      <td>274</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1272</td>\n",
              "      <td>1149</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     2\n",
              "Actual attacks               \n",
              "0                  9437   274\n",
              "2                  1272  1149"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 50
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "pHs6iaNZ6iNI",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EsV-50ki6Hm_",
        "colab_type": "code",
        "outputId": "8599b605-b482-4b5a-db61-83b06fa6b8c6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_R2L_pred=clf_KNN_R2L.predict(X_R2L_test)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_R2L_test, Y_R2L_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>3</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9706</td>\n",
              "      <td>5</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>2883</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0  3\n",
              "Actual attacks            \n",
              "0                  9706  5\n",
              "3                  2883  2"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 51
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zPwGBEeH6mji",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RgO_fAYu6HvD",
        "colab_type": "code",
        "outputId": "e18dc016-7081-4f51-be63-e4467b2523e0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_U2R_pred=clf_KNN_U2R.predict(X_U2R_test)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_U2R_test, Y_U2R_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>4</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9711</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>65</td>\n",
              "      <td>2</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0  4\n",
              "Actual attacks            \n",
              "0                  9711  0\n",
              "4                    65  2"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 52
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "g9pZadi86vhE",
        "colab_type": "text"
      },
      "source": [
        "**Cross Validation: Accuracy, Precision, Recall, F-measure**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bQU5VuWl6p9L",
        "colab_type": "text"
      },
      "source": [
        "**DoS**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "SJ4QM5TV6H39",
        "colab_type": "code",
        "outputId": "fc8632e0-33e3-43fb-d53e-f258fc755a69",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn import metrics\n",
        "accuracy = cross_val_score(clf_KNN_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_KNN_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='precision')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_KNN_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='recall')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_KNN_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='f1')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99715 (+/- 0.00278)\n",
            "Precision: 0.99678 (+/- 0.00383)\n",
            "Recall: 0.99665 (+/- 0.00344)\n",
            "F-measure: 0.99672 (+/- 0.00320)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "a7GfPZaI6zTy",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ftIa3IDi6zln",
        "colab_type": "code",
        "outputId": "ef1a751b-e4b9-41f7-a0e4-42c8e47923b7",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_KNN_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_KNN_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_KNN_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_KNN_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99077 (+/- 0.00403)\n",
            "Precision: 0.98606 (+/- 0.00674)\n",
            "Recall: 0.98508 (+/- 0.01137)\n",
            "F-measure: 0.98553 (+/- 0.00645)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-W_1Iw2P6z2r",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "8O_VI52o60Ih",
        "colab_type": "code",
        "outputId": "b8ecbc62-8939-4a62-dc55-20a86d8463a8",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_KNN_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_KNN_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_KNN_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_KNN_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.96737 (+/- 0.00729)\n",
            "Precision: 0.95311 (+/- 0.01273)\n",
            "Recall: 0.95484 (+/- 0.01327)\n",
            "F-measure: 0.95389 (+/- 0.01030)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qrkB_Fbn60Zm",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "g4q8MlDu60jn",
        "colab_type": "code",
        "outputId": "11fc94f3-6721-4058-be22-8234b5f88f5d",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_KNN_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_KNN_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_KNN_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_KNN_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99703 (+/- 0.00281)\n",
            "Precision: 0.93282 (+/- 0.14488)\n",
            "Recall: 0.84835 (+/- 0.17662)\n",
            "F-measure: 0.87754 (+/- 0.11386)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "bvLJSVKmBNDV",
        "colab_type": "text"
      },
      "source": [
        "# SVM\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FywpAGEvBPLo",
        "colab_type": "code",
        "outputId": "9c6759e3-2fca-49f4-d969-21ba4c9fc55c",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "from sklearn.svm import SVC\n",
        "\n",
        "clf_SVM_DoS=SVC(kernel='linear', C=1.0, random_state=0)\n",
        "clf_SVM_Probe=SVC(kernel='linear', C=1.0, random_state=0)\n",
        "clf_SVM_R2L=SVC(kernel='linear', C=1.0, random_state=0)\n",
        "clf_SVM_U2R=SVC(kernel='linear', C=1.0, random_state=0)\n",
        "\n",
        "clf_SVM_DoS.fit(X_DoS, Y_DoS.astype(int))\n",
        "clf_SVM_Probe.fit(X_Probe, Y_Probe.astype(int))\n",
        "clf_SVM_R2L.fit(X_R2L, Y_R2L.astype(int))\n",
        "clf_SVM_U2R.fit(X_U2R, Y_U2R.astype(int))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,\n",
              "    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',\n",
              "    kernel='linear', max_iter=-1, probability=False, random_state=0,\n",
              "    shrinking=True, tol=0.001, verbose=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 57
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "71tRFjJGBiu8",
        "colab_type": "text"
      },
      "source": [
        "**DoS**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kABBJljaBi6v",
        "colab_type": "code",
        "outputId": "718b7df7-b939-4244-b0ec-278ec9bdec71",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_DoS_pred=clf_SVM_DoS.predict(X_DoS_test)\n",
        "\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9455</td>\n",
              "      <td>256</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1359</td>\n",
              "      <td>6101</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     1\n",
              "Actual attacks               \n",
              "0                  9455   256\n",
              "1                  1359  6101"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 58
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iuj91846Hd3D",
        "colab_type": "code",
        "outputId": "dea77b6e-0777-43d2-dc00-593733973a81",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_Probe_pred=clf_SVM_Probe.predict(X_Probe_test)\n",
        "# Create confusion matrix\n",
        "\n",
        "pd.crosstab(Y_Probe_test, Y_Probe_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>2</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9576</td>\n",
              "      <td>135</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1285</td>\n",
              "      <td>1136</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     2\n",
              "Actual attacks               \n",
              "0                  9576   135\n",
              "2                  1285  1136"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 59
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lAVZ2ZTvHgq1",
        "colab_type": "code",
        "outputId": "3e27c583-a19a-4fac-eca1-95f221428bc9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_R2L_pred=clf_SVM_R2L.predict(X_R2L_test)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_R2L_test, Y_R2L_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>3</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9639</td>\n",
              "      <td>72</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>2737</td>\n",
              "      <td>148</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0    3\n",
              "Actual attacks              \n",
              "0                  9639   72\n",
              "3                  2737  148"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 60
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iJusMfB6Hj1m",
        "colab_type": "code",
        "outputId": "17b7ec80-5acf-42a7-8033-d83074ee2c6f",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_U2R_pred=clf_SVM_U2R.predict(X_U2R_test)\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_U2R_test, Y_U2R_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>4</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9710</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>67</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0  4\n",
              "Actual attacks            \n",
              "0                  9710  1\n",
              "4                    67  0"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 61
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FHJNIbkSDdrJ",
        "colab_type": "text"
      },
      "source": [
        "**DoS**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "X4MfSJPbVs32",
        "colab_type": "code",
        "outputId": "cdf5d0b7-3f28-4f29-d32e-8bfee96ed7a0",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn import metrics\n",
        "accuracy = cross_val_score(clf_SVM_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_SVM_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='precision')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_SVM_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='recall')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_SVM_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='f1')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99371 (+/- 0.00375)\n",
            "Precision: 0.99107 (+/- 0.00785)\n",
            "Recall: 0.99450 (+/- 0.00388)\n",
            "F-measure: 0.99278 (+/- 0.00428)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "IQqgT2jwDhn5",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9P36p-moVtBa",
        "colab_type": "code",
        "outputId": "1a7a545d-49db-4f8c-d015-f1ea6bb916eb",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_SVM_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_SVM_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_SVM_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_SVM_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.98450 (+/- 0.00525)\n",
            "Precision: 0.96907 (+/- 0.01029)\n",
            "Recall: 0.98365 (+/- 0.00686)\n",
            "F-measure: 0.97613 (+/- 0.00799)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EOEYHrRYDkaL",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1EUHxx_FVtI-",
        "colab_type": "code",
        "outputId": "699a600e-d387-4bc1-9eb9-8c64dbc132ea",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_SVM_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_SVM_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_SVM_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_SVM_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.96793 (+/- 0.00738)\n",
            "Precision: 0.94854 (+/- 0.00992)\n",
            "Recall: 0.96264 (+/- 0.01386)\n",
            "F-measure: 0.95529 (+/- 0.01046)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ghjSj7ucDoGE",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Qcz_NY5mVtQo",
        "colab_type": "code",
        "outputId": "befd07f4-425c-413a-fbde-f5cbabc7d4e4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_SVM_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_SVM_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_SVM_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_SVM_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99652 (+/- 0.00245)\n",
            "Precision: 0.91988 (+/- 0.15114)\n",
            "Recall: 0.83981 (+/- 0.17847)\n",
            "F-measure: 0.85918 (+/- 0.10713)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RpavTYvvA9Tm",
        "colab_type": "text"
      },
      "source": [
        "# Ensemble Learning"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IiEH9DP8BBRf",
        "colab_type": "code",
        "outputId": "825cf844-c056-4d35-98f5-77dcf339cc99",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 493
        }
      },
      "source": [
        "from sklearn.ensemble import VotingClassifier\n",
        "\n",
        "clf_voting_DoS = VotingClassifier(estimators=[('rf', clf_DoS), ('knn', clf_KNN_DoS), ('svm', clf_SVM_DoS)], voting='hard')\n",
        "clf_voting_Probe = VotingClassifier(estimators=[('rf', clf_Probe), ('knn', clf_KNN_Probe), ('svm', clf_SVM_Probe)], voting='hard')\n",
        "clf_voting_R2L = VotingClassifier(estimators=[('rf', clf_R2L), ('knn', clf_KNN_R2L), ('svm', clf_SVM_R2L)], voting='hard')\n",
        "clf_voting_U2R = VotingClassifier(estimators=[('rf', clf_U2R), ('knn', clf_KNN_U2R), ('svm', clf_SVM_U2R)], voting='hard')\n",
        "\n",
        "clf_voting_DoS.fit(X_DoS, Y_DoS.astype(int))\n",
        "clf_voting_Probe.fit(X_Probe, Y_Probe.astype(int))\n",
        "clf_voting_R2L.fit(X_R2L, Y_R2L.astype(int))\n",
        "clf_voting_U2R.fit(X_U2R, Y_U2R.astype(int))\n"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "VotingClassifier(estimators=[('rf',\n",
              "                              RandomForestClassifier(bootstrap=True,\n",
              "                                                     class_weight=None,\n",
              "                                                     criterion='gini',\n",
              "                                                     max_depth=None,\n",
              "                                                     max_features='auto',\n",
              "                                                     max_leaf_nodes=None,\n",
              "                                                     min_impurity_decrease=0.0,\n",
              "                                                     min_impurity_split=None,\n",
              "                                                     min_samples_leaf=1,\n",
              "                                                     min_samples_split=2,\n",
              "                                                     min_weight_fraction_leaf=0.0,\n",
              "                                                     n_estimators=10, n_jobs=2,\n",
              "                                                     oob_score=False,\n",
              "                                                     random_state=None,\n",
              "                                                     verbose=0,...\n",
              "                                                   metric_params=None,\n",
              "                                                   n_jobs=None, n_neighbors=5,\n",
              "                                                   p=2, weights='uniform')),\n",
              "                             ('svm',\n",
              "                              SVC(C=1.0, cache_size=200, class_weight=None,\n",
              "                                  coef0=0.0, decision_function_shape='ovr',\n",
              "                                  degree=3, gamma='auto_deprecated',\n",
              "                                  kernel='linear', max_iter=-1,\n",
              "                                  probability=False, random_state=0,\n",
              "                                  shrinking=True, tol=0.001, verbose=False))],\n",
              "                 flatten_transform=True, n_jobs=None, voting='hard',\n",
              "                 weights=None)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 68
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "HMhaEDXOCtpk",
        "colab_type": "text"
      },
      "source": [
        "**DoS**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "PcnHXWiNCvL2",
        "colab_type": "code",
        "outputId": "3e43f8a7-d621-46a5-e8f4-d7fcdd1297c5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_DoS_pred=clf_voting_DoS.predict(X_DoS_test)\n",
        "\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_DoS_test, Y_DoS_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9607</td>\n",
              "      <td>104</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1797</td>\n",
              "      <td>5663</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     1\n",
              "Actual attacks               \n",
              "0                  9607   104\n",
              "1                  1797  5663"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 69
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "2LBoLeEsC8pm",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eUnAparzC-B3",
        "colab_type": "code",
        "outputId": "cf2be410-189f-4b68-fe12-c212d6cf99c5",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_Probe_pred=clf_voting_Probe.predict(X_Probe_test)\n",
        "\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_Probe_test, Y_Probe_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>2</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9479</td>\n",
              "      <td>232</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1182</td>\n",
              "      <td>1239</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0     2\n",
              "Actual attacks               \n",
              "0                  9479   232\n",
              "2                  1182  1239"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 70
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Dx1S4pNfDIV8",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "flUiIlHJDKYe",
        "colab_type": "code",
        "outputId": "ffda78ee-9119-4c5d-a19a-b45780c9b25b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_R2L_pred=clf_voting_R2L.predict(X_R2L_test)\n",
        "\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_R2L_test, Y_R2L_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "      <th>3</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9711</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>2884</td>\n",
              "      <td>1</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0  3\n",
              "Actual attacks            \n",
              "0                  9711  0\n",
              "3                  2884  1"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 71
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "A46mT-XBDQ2G",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3u9wU5OcDSF3",
        "colab_type": "code",
        "outputId": "495dcd2b-26a5-442c-c755-607deab293d9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 142
        }
      },
      "source": [
        "Y_U2R_pred=clf_voting_U2R.predict(X_U2R_test)\n",
        "\n",
        "# Create confusion matrix\n",
        "pd.crosstab(Y_U2R_test, Y_U2R_pred, rownames=['Actual attacks'], colnames=['Predicted attacks'])"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "execute_result",
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
              "      <th>Predicted attacks</th>\n",
              "      <th>0</th>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>Actual attacks</th>\n",
              "      <th></th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>9711</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>67</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "Predicted attacks     0\n",
              "Actual attacks         \n",
              "0                  9711\n",
              "4                    67"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 72
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ojnaDLQZDt_G",
        "colab_type": "text"
      },
      "source": [
        "**DoS**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VZf04XKLDrvL",
        "colab_type": "code",
        "outputId": "477f8c9d-cb26-47eb-a4d8-8fdb8f2c3f53",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "from sklearn.model_selection import cross_val_score\n",
        "from sklearn import metrics\n",
        "accuracy = cross_val_score(clf_voting_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_voting_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='precision')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_voting_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='recall')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_voting_DoS, X_DoS_test, Y_DoS_test, cv=10, scoring='f1')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99802 (+/- 0.00196)\n",
            "Precision: 0.99852 (+/- 0.00280)\n",
            "Recall: 0.99705 (+/- 0.00289)\n",
            "F-measure: 0.99772 (+/- 0.00256)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qlGDiSptD2nJ",
        "colab_type": "text"
      },
      "source": [
        "**Probe**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "n83y5PJbD3ni",
        "colab_type": "code",
        "outputId": "cb6d105c-4e19-4058-81bf-1ab70c98541b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 68
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_voting_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_voting_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_voting_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_voting_Probe, X_Probe_test, Y_Probe_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-mesaure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99283 (+/- 0.00411)\n",
            "Precision: 0.98765 (+/- 0.00767)\n",
            "Recall: 0.98952 (+/- 0.00744)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wv-xS9-FEQh-",
        "colab_type": "text"
      },
      "source": [
        "**R2L**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tnKA7-GBESPW",
        "colab_type": "code",
        "outputId": "ad870774-cb62-4611-8427-9bb38ca59ab4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_voting_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_voting_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_voting_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_voting_R2L, X_R2L_test, Y_R2L_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.97213 (+/- 0.00653)\n",
            "Precision: 0.95811 (+/- 0.00810)\n",
            "Recall: 0.96433 (+/- 0.01313)\n",
            "F-measure: 0.96029 (+/- 0.00707)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "moAQ-pd9EZAi",
        "colab_type": "text"
      },
      "source": [
        "**U2R**"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "74Lwk4bEEaex",
        "colab_type": "code",
        "outputId": "826f1311-1878-4633-af87-174acd9a258b",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 85
        }
      },
      "source": [
        "accuracy = cross_val_score(clf_voting_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='accuracy')\n",
        "print(\"Accuracy: %0.5f (+/- %0.5f)\" % (accuracy.mean(), accuracy.std() * 2))\n",
        "precision = cross_val_score(clf_voting_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='precision_macro')\n",
        "print(\"Precision: %0.5f (+/- %0.5f)\" % (precision.mean(), precision.std() * 2))\n",
        "recall = cross_val_score(clf_voting_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='recall_macro')\n",
        "print(\"Recall: %0.5f (+/- %0.5f)\" % (recall.mean(), recall.std() * 2))\n",
        "f = cross_val_score(clf_voting_U2R, X_U2R_test, Y_U2R_test, cv=10, scoring='f1_macro')\n",
        "print(\"F-measure: %0.5f (+/- %0.5f)\" % (f.mean(), f.std() * 2))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Accuracy: 0.99765 (+/- 0.00243)\n",
            "Precision: 0.94515 (+/- 0.11771)\n",
            "Recall: 0.86144 (+/- 0.18332)\n",
            "F-measure: 0.90643 (+/- 0.08475)\n"
          ],
          "name": "stdout"
        }
      ]
    }
  ]
}