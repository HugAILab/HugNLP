from transformers import PLBartConfig
import sys
sys.path.append("./")
sys.path.append("../")
sys.path.append("../../")
from models import CODE_MODEL_CLASSES
from models import TOKENIZER_CLASSES

class HugCodeAPI:
    def __init__(self, model_type, hugcode_model_name_or_path) -> None:
        if model_type not in CODE_MODEL_CLASSES["code_cls"].keys():
            raise KeyError(
                "You must choose one of the following model: {}".format(
                    ", ".join(
                        list(CODE_MODEL_CLASSES["code_cls"].
                             keys()))))
        self.model_type = model_type
        self.model = CODE_MODEL_CLASSES["code_cls"][
            self.model_type].from_pretrained(hugcode_model_name_or_path)
        self.tokenizer = TOKENIZER_CLASSES[self.model_type].from_pretrained(
            hugcode_model_name_or_path)
        self.eos_token_id = 2
        self.max_source_length = 512
        self.max_target_length = 512

    def request(self, func1: str, func2: str):
            assert text is not None and entity_type is not None
            if relation is None:
                instruction = "找到文章中所有【{}】类型的实体？文章：【{}】".format(entity_type, text)
                pre_len = 21 - 2 + len(entity_type)
            else:
                instruction = "找到文章中【{}】的【{}】？文章：【{}】".format(
                    entity_type, relation, text)
                pre_len = 19 - 4 + len(entity_type) + len(relation)

            inputs = self.tokenizer(instruction,
                                    max_length=self.max_seq_length,
                                    padding="max_length",
                                    return_tensors="pt",
                                    return_offsets_mapping=True)

            examples = {
                "content": [instruction],
                "offset_mapping": inputs["offset_mapping"]
            }

            batch_input = {
                "input_ids": inputs["input_ids"],
                "token_type_ids": inputs["token_type_ids"],
                "attention_mask": inputs["attention_mask"],
            }

            outputs = self.model(**batch_input)

            probs, indices = outputs["topk_probs"], outputs["topk_indices"]
            predictions, topk_predictions = self.get_predict_result(
                probs, indices, examples=examples)

            return clone_probability
# 从本地路径加载模型和配置

model = CODE_MODEL_CLASSES["code_cls"]["plbart"].from_pretrained("/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1")#("/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1/pytorch_model.bin")
# tokenizer = TOKENIZER_CLASSES["plbart"].from_pretrained("/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/checkpoint-best-f1")#("/root/autodl-tmp/CodePrompt/data/huggingface_models/plbart-base/sentencepiece.bpe.model")


if __name__ == "__main__":
    from applications.code.HugCode.clone_api import HugCodeAPI
    model_type = "plbart"
    hugie_model_name_or_path = "/root/autodl-tmp/code/CodePrompt/save_models/clone/plbart/ckpt_test/"
    hugie = HugCodeAPI(model_type, hugie_model_name_or_path)

    ## JAVA code clone detection
    func1="""
    public String getData(DefaultHttpClient httpclient) {
        try {
            HttpGet get = new HttpGet("http://3dforandroid.appspot.com/api/v1/note");
            get.setHeader("Content-Type", "application/json");
            get.setHeader("Accept", "*/*");
            HttpResponse response = httpclient.execute(get);
            HttpEntity entity = response.getEntity();
            InputStream instream = entity.getContent();
            responseMessage = read(instream);
            if (instream != null) instream.close();
        } catch (ClientProtocolException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return responseMessage;
    }
    """
    func2="""
    public static void copyFile(File in, File out) throws Exception {
        FileChannel sourceChannel = new FileInputStream(in).getChannel();
        FileChannel destinationChannel = new FileOutputStream(out).getChannel();
        sourceChannel.transferTo(0, sourceChannel.size(), destinationChannel);
        sourceChannel.close();
        destinationChannel.close();
    }
    """
    clone_probability = hugie.request(func1, func2)
    print("clone_probability:{}".format(clone_probability))
    print("\n\n")

    ## JAVA code clone detection
    func1="""
    public static void copyFile(File source, File dest) throws IOException {
        FileChannel in = null, out = null;
        try {
            in = new FileInputStream(source).getChannel();
            out = new FileOutputStream(dest).getChannel();
            in.transferTo(0, in.size(), out);
        } catch (FileNotFoundException fnfe) {
            Log.debug(fnfe);
        } finally {
            if (in != null) in.close();
            if (out != null) out.close();
        }
    }
    """

    func2="""
    public static void copyFile(File from, File to) throws IOException {
        if (from.isDirectory()) {
            if (!to.exists()) {
                to.mkdir();
            }
            File[] children = from.listFiles();
            for (int i = 0; i < children.length; i++) {
                if (children[i].getName().equals(".") || children[i].getName().equals("..")) {
                    continue;
                }
                if (children[i].isDirectory()) {
                    File f = new File(to, children[i].getName());
                    copyFile(children[i], f);
                } else {
                    copyFile(children[i], to);
                }
            }
        } else if (from.isFile() && (to.isDirectory() || to.isFile())) {
            if (to.isDirectory()) {
                to = new File(to, from.getName());
            }
            FileInputStream in = new FileInputStream(from);
            FileOutputStream out = new FileOutputStream(to);
            byte[] buf = new byte[32678];
            int read;
            while ((read = in.read(buf)) > -1) {
                out.write(buf, 0, read);
            }
            closeStream(in);
            closeStream(out);
        }
    }
    """
    clone_probability = hugie.request(func1, func2)
    print("clone_probability:{}".format(clone_probability))
    print("\n\n")

"""
    Output results:

    entity_type:国家
predictions:
{0: ["塔吉克斯坦"]}
predictions:
{0: [{"answer": "塔吉克斯坦", "prob": 0.9999997615814209, "pos": [(tensor(57), tensor(62))]}]}



entity:塔吉克斯坦地震, relation:震源深度
predictions:
{0: ["10公里"]}
predictions:
{0: [{"answer": "10公里", "prob": 0.999994158744812, "pos": [(tensor(80), tensor(84))]}]}



entity:塔吉克斯坦地震, relation:震源位置
predictions:
{0: ["10公里", "距我国边境线最近约82公里", "北纬37.98度，东经73.29度", "北纬37.98度，东经73.29度，距我国边境线最近约82公里"]}
predictions:
{0: [{"answer": "10公里", "prob": 0.9895901083946228, "pos": [(tensor(80), tensor(84))]}, {"answer": "距我国边境线最近约82公里", "prob": 0.8584909439086914, "pos": [(tensor(107), tensor(120))]}, {"answer": "北纬37.98度，东经73.29度", "prob": 0.7202121615409851, "pos": [(tensor(89), tensor(106))]}, {"answer": "北纬37.98度，东经73.29度，距我国边境线最近约82公里", "prob": 0.11628123372793198, "pos": [(tensor(89), tensor(120))]}]}



entity:塔吉克斯坦地震, relation:时间
predictions:
{0: ["2月23日8时37分"]}
predictions:
{0: [{"answer": "2月23日8时37分", "prob": 0.9999995231628418, "pos": [(tensor(49), tensor(59))]}]}



entity:塔吉克斯坦地震, relation:影响
predictions:
{0: ["新疆喀什等地震感强烈"]}
predictions:
{0: [{"answer": "新疆喀什等地震感强烈", "prob": 0.9525265693664551, "pos": [(tensor(123), tensor(133))]}]}

    """
