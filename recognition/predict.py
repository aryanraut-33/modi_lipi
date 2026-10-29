"""Predict one segmented glyph, or import Recognizer for Phase 4 integration."""
import argparse
import json
import torch
from PIL import Image
from model import ARCH, build_model, prepare_image, tensor_image


class Recognizer:
    def __init__(self, checkpoint, device='cpu'):
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if data['arch'] != ARCH or data['kind'] != 'modi':
            raise ValueError('Expected a trained Modi checkpoint')
        self.labels = data['labels']
        self.device = torch.device(device)
        self.model = build_model(len(self.labels)).to(self.device)
        self.model.load_state_dict(data['state_dict'])
        self.model.eval()

    @torch.no_grad()
    def predict(self, image, topk=5):
        """Accept PIL image; for pipeline glyphs use Image.fromarray(glyph.full_glyph_img)."""
        tensor = tensor_image(prepare_image(image)).unsqueeze(0).to(self.device)
        scores, ids = self.model(tensor).softmax(1)[0].topk(min(max(1,topk),len(self.labels)))
        return [{'label': self.labels[i], 'score': float(score)} for score,i in zip(scores.cpu(),ids.cpu().tolist())]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()
    with Image.open(args.image) as image:
        print(json.dumps(Recognizer(args.checkpoint).predict(image), ensure_ascii=False, indent=2))
