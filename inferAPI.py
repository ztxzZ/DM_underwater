import torch
from core import metrics as Metrics
import data as Data
import model as Model
import time
import core.logger as Logger
import logging
import os
from argparse import Namespace

class DiffusionModelInfer:
    def __init__(self, config_path='config/underwater.json', gpu_ids=None):
        # 创建模拟的argparse参数对象
        args = Namespace(
            config=config_path,
            phase='val',
            gpu_ids=gpu_ids,
            debug=False,
            enable_wandb=False,
            log_infer=False
        )
        
        # 完整配置初始化流程
        self.opt = Logger.parse(args)
        self.opt = Logger.dict_to_nonedict(self.opt)
        
        # 硬件设置
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化日志系统
        Logger.setup_logger(None, self.opt['path']['log'], 'train', level=logging.INFO, screen=True)
        self.logger = logging.getLogger('base')
        
        # 数据集初始化（完整流程）
        val_dataset_opt = self.opt['datasets']['val']
        self.val_set = Data.create_dataset(val_dataset_opt, 'val')
        self.val_loader = Data.create_dataloader(self.val_set, val_dataset_opt, 'val')
        
        # 模型初始化
        self.diffusion = Model.create_model(self.opt).to(self.device)
        self.diffusion.set_new_noise_schedule(
            self.opt['model']['beta_schedule']['val'], 
            schedule_phase='val'
        )
        
        # 结果目录准备
        self.result_path = self.opt['path']['results']
        os.makedirs(self.result_path, exist_ok=True)

    def infer(self, input_image, save_results=True):
        """
        完整推理流程
        :param input_image: PIL.Image输入图像
        :param save_results: 是否保存结果文件
        :return: 包含结果数据和路径的字典
        """
        # 预处理
        transform = Data.create_transform(self.opt['datasets']['val']['dataroot_GT'])
        input_tensor = transform(input_image).unsqueeze(0).to(self.device)
        
        # 推理执行
        self.diffusion.eval()
        with torch.no_grad():
            self.diffusion.feed_data({'HR': input_tensor})
            start_time = time.time()
            self.diffusion.test(continous=True)
            inference_time = time.time() - start_time

        # 结果处理
        visuals = self.diffusion.get_current_visuals(need_LR=False)
        results = {
            'hr': Metrics.tensor2img(visuals['HR']),
            'output': Metrics.tensor2img(visuals['SR'][-1]),
            'process': [Metrics.tensor2img(img) for img in visuals['SR']],
            'inference_time': inference_time
        }

        # 结果保存
        if save_results:
            idx = len(os.listdir(self.result_path)) // 5 + 1
            Metrics.save_img(results['hr'], f'{self.result_path}/{idx}_hr.png')
            Metrics.save_img(results['output'], f'{self.result_path}/{idx}_sr.png')
            Metrics.save_img(Metrics.tensor2img(visuals['SR']), 
                           f'{self.result_path}/{idx}_sr_process.png')
            results['save_path'] = self.result_path

        return results

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    inferer = DiffusionModelInfer(gpu_ids='0')
    test_image = Image.open("test.jpg").convert('RGB')
    results = inferer.infer(test_image)
    output_image = Image.fromarray(results['output'])