#!/bin/bash

mkdir -p ./output

# 読ませたい文章をutf-8でtext.txtに書き出す
echo -n "ディープラーニングは万能薬ではありません" > ./output/text.txt

curl -s \
    -X POST \
    "127.0.0.1:50021/audio_query?speaker=17" \
    --get --data-urlencode text@./output/text.txt \
    > ./output/query.json

cat ./output/query.json | grep -o -E "\"kana\":\".*\""
# 結果... "kana":"ディ'イプ/ラ'アニングワ/バンノオヤクデワアリマセ'ン"

# "ディイプラ'アニングワ/バンノ'オヤクデワ/アリマセ'ン"と読ませたいので、
# is_kana=trueをつけてイントネーションを取得しnewphrases.jsonに保存
echo -n "ディイプラ'アニングワ/バンノ'オヤクデワ/アリマセ'ン" > ./output/kana.txt
curl -s \
    -X POST \
    "127.0.0.1:50021/accent_phrases?speaker=1&is_kana=true" \
    --get --data-urlencode text@./output/kana.txt \
    > ./output/newphrases.json

# query.jsonの"accent_phrases"の内容をnewphrases.jsonの内容に置き換える
cat ./output/query.json | sed -e "s/\[{.*}\]/$(cat ./output/newphrases.json)/g" > ./output/newquery.json

curl -s \
    -H "Content-Type: application/json" \
    -X POST \
    -d @./output/newquery.json \
    "127.0.0.1:50021/synthesis?speaker=17" \
    > ./output/audio.wav
