# kamattechan

AIｽﾀｯｸﾁｬﾝというチャットボットアプリケーションです。

## Docker環境の使用方法

このプロジェクトでは、VOICEVOXエンジンをDockerで実行するための設定が含まれています。


### VOICEVOXエンジンの起動

以下のコマンドを実行して、VOICEVOXエンジンを起動します：

```bash
docker-compose up -d
```

これにより、VOICEVOXエンジンがバックグラウンドで起動し、ポート50021でリッスンします。

### VOICEVOXエンジンの停止

以下のコマンドを実行して、VOICEVOXエンジンを停止します：

```bash
docker-compose down
```

### ログの確認

以下のコマンドを実行して、VOICEVOXエンジンのログを確認できます：

```bash
docker-compose logs -f
```

## APIエンドポイント

VOICEVOXエンジンが起動すると、以下のURLでAPIにアクセスできます：

- http://127.0.0.1:50021
