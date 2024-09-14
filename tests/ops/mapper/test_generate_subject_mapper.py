import sys
import os
# 设置路径
root_path="/mnt/pfs/jinfeng_team/SFT/qinxiuyuan/workplace/dataprocess/data-juicer"
sys.path.insert(0,root_path)


# export TOKENIZERS_PARALLELISM=false python /mnt/pfs/jinfeng_team/SFT/qinxiuyuan/workplace/dataprocess/data-juicer/tests/ops/mapper/test_generate_subject_tag_mapper.py

import unittest
import json
# 加载op中的算子
from data_juicer.ops.mapper.generate_subject_mapper import GenerateSubjectMapper
from data_juicer.utils.unittest_utils import (SKIPPED_TESTS,
                                              DataJuicerTestCaseBase)

# Skip tests for this OP in the GitHub actions due to disk space limitation.
# These tests have been tested locally.
@SKIPPED_TESTS.register_module()
class GenerateSubjectMapperTest(DataJuicerTestCaseBase):

    text_key = 'text'

    # 初始化op
    def _run_generate_subject(self, enable_vllm=False):
        op = GenerateSubjectMapper(
            hf_model='/mnt/pfs/jinfeng_team/IT/gwxie/modelFiles/qwen2/train_3_head/train_qwen2_4_head_2048token_pretrain_v2_3/checkpoint-1242',
            tensor_parallel_size=1,
            max_input_length=512,
            enable_vllm=enable_vllm
        )

        data_file='/mnt/pfs/jinfeng_team/SFT/qinxiuyuan/workplace/dataprocess/data-juicer/demos/data/demo-dataset-subject.jsonl'
        # 加载数据，这里可以适配不同的格式
        from data_juicer.format.json_formatter import JsonFormatter
        dataset = JsonFormatter(data_file).load_dataset()
        # print(dataset)

        dataset = dataset.map(op.process)

        for item in dataset:            
            out_sample = json.loads(item[self.text_key])
            print(f'Output sample: {out_sample}')
    
    # 单元测试1
    # def test_generate_subject(self):
        # self._run_generate_instruction()

    # 单元测试2
    def test_generate_subject_vllm(self):
        self._run_generate_subject(enable_vllm=True)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    os.environ["CUDA_LAUNCH_BLOCKING"]='1'
    unittest.main()
