lightgcn을 recbole을 이용해서 결과를 얻고자 합니다.


path
- 본인의 train/test data경로에 맞춰주시면됩니다.

추가할 데이터는 elapsed입니다. 추가적으로 넣고싶으면 data에 추가한 후 config에 적용해주시기 바랍니다. 
- elasped
    - 문제 푼 시간 


yaml
- recbole의 config 파일입니다.
- ~~FIELD : 각각의 feature를 넣습니다
    - 추가하고 싶으면 임의로 이름을 지정해주시고 feature이름과 매칭해주시기 바랍니다.
- load_col
    - 지정한 feature를 여기서 받아들입니다.

- test도 train과 동일하게 지정해주십시오.

interaction
- interaction을 지칭하는 inter, yaml을 불러옵니다.
- inter_table : yaml에서 지정한 col을 여기에 추가적으로 넣어줍니다.

run_recbole
- 설정한 config를 바탕으로 recbole을 실행시킵니다.