import re
import os

class EENet:
    def __init__(self, file):
        self.file = file
        self.backbone = []
        self.exits = []
        self.exit_point = []
        self.threshold = [0.9, 0.9, 0.9]
        self.wtsName = '.wts'

        self.workspace = '128_MiB'
        self.warmup = 0
        self.fix_pee = False
        self.nsight = False
        self.print_individual = False
        
        self.engines = 1
        self.interDims = 0
        self.binding_sizes = []

        self.HLayerFusion = False
        self.multistream = False
        self.multithread = False

        self.mallocBindings = ''
        self.freeBindings = ''

        self._parse_model()
        self.LF_bkpt = self.exit_point[:]
        self.BB_bkpt = self.exit_point[:]
        
    def _parse_model(self):
        f = open(self.file, 'r')
        data = f.readlines()
        f.close()

        self.exit_point.append(0)
        keywords = ['ModuleList', 'Sequential']
        last_indent = 0
        layer_name = []
        last_exit_point = 0
        ctr = 0

        for i, line in enumerate(data):
            line = line[:-1]
            if i == 0:
                self.wtsName = line[:-1] + '.wts'
                _, self.model_name, self.dataset = line[:-1].split('_')
                last_indent = len(line) - len(line.strip())
            elif ctr > 0:
                ctr -= 1
            else:
                indent = len(line) - len(line.strip())
                line = line.strip().split(':')
                line[0] = line[0][1:-1]

                if indent > last_indent:
                    layer_name.append(line[0])
                elif indent == last_indent:
                    layer_name.pop()
                    layer_name.append(line[0])
                else:
                    layer_name.pop()
                    last_indent = indent
                    continue
                
                line[1] = line[1].strip()
                if not line[1][:-1] in keywords:
                    layer = line[1]
                    if line[1][:-1] == 'ResBlock':
                        ctr = 1
                        while True:
                            layer += '$' + data[i+ctr].strip()
                            if ('(bn2)' in data[i+ctr]):
                                layer += '\')'
                                layer = layer[:layer.index('$')] + '\'' + layer[layer.index('$')+1:]
                                ctr += 1
                                break
                            ctr += 1
                    if layer_name[0] == 'backbone':
                        self.backbone.append(f"{layer[:-1]}, wts=\'{'.'.join(layer_name)}\'{layer[-1:]}")
                        if last_exit_point != int(layer_name[1]):
                            self.exit_point.append(self.exit_point[-1])
                        self.exit_point[-1] += 1
                        last_exit_point = int(layer_name[1])
                    else:
                        if indent > last_indent:
                            self.exits.append([])
                        self.exits[-1].append(f"{layer[:-1]}, wts=\'{'.'.join(layer_name)}\'{layer[-1:]}")
                
                last_indent = indent

        self.backbone.append(self.exits.pop()[0])
        del self.exit_point[-1]
        self.backbone.append('Softmax()')
        for exit in self.exits:
            exit.append('Softmax()')

        for i, layer in enumerate(self.backbone):
            if 'Flatten' in layer:
                self.backbone[i] = f"Flatten({re.search('in_features=(.*), out', self.backbone[i+1]).group(1)})"
            if 'Dropout' in layer:
                del self.backbone[i]

        for exit in self.exits:
            for i, layer in enumerate(exit):
                if 'Flatten' in layer:
                    exit[i] = f"Flatten({re.search('in_features=(.*), out', exit[i+1]).group(1)})"
                if 'Dropout' in layer:
                    del exit[i]
        for i in range (len(self.backbone)):
            self.backbone[i] = [self.backbone[i], f'bb_{self.backbone[i][:self.backbone[i].index("(")]}_{i}']
        for i in range(len(self.exits)):
            for j in range(len(self.exits[i])):
                self.exits[i][j] = [self.exits[i][j], f'ee{i}_{self.exits[i][j][:self.exits[i][j].index("(")]}_{j}']
    
    def set_threshold(self, t):
        self.threshold = t
    
    def add_code(self, code, eol=1):
        self.trt_code += code + '\n' * eol

    def write_cpp(self):
        f = open('EENet.cpp', "w")
        f.write(self.trt_code)
        f.close()
    
    def compile(self):
        os.system("make -j24")
    
    def execute(self):
        os.system("./../../bin/EENet")

    def gen_backbone(self):
        self.engines = 1
        self.gen_trt()
        self.add_code(self.defBegin().format(network = '0', dims = "Dims4{1, mParams.inputC, mParams.inputH, mParams.inputW}"))
        self.gen_layers(self.backbone, '0', 'data')
        self.add_code(self.addOutput('0', self.backbone[-1][1]))
        self.add_code(self.defEnd().format(network = '0', workspace = self.workspace))
    
    def gen_baseline(self):
        self.engines = 1
        self.gen_trt()
        self.add_code(self.defBegin().format(network = '0', dims = "Dims4{1, mParams.inputC, mParams.inputH, mParams.inputW}"))
        self.add_code("    const float T = 0.0125f;")
        self.add_code("    const Weights TH{DataType::kFLOAT, &T, 1};")

        self.gen_layers(self.backbone, '0', 'data')
        for i in range (len(self.exits)):
            self.gen_layers(self.exits[i], '0', self.backbone[self.exit_point[i]-1][1] + '->getOutput(0)')
        for i in reversed(range(len(self.exits))):
            self.add_code(self.selectLayer(i, self.exits[i][-1][1], self.backbone[-1][1] if i == len(self.exits)-1 else f'select{i+1}'))
        self.add_code(self.addOutput('0', 'select0'))
        self.add_code(self.defEnd().format(network = '0', workspace = self.workspace))

    def gen_optimize(self):
        self.gen_bindings()
        self.gen_trt()
        self.LF_bkpt.insert(0, 0)
        self.BB_bkpt.insert(0, 0)
        if not self.multistream:
            for i in range (self.engines-1):
                if i == 0:
                    self.add_code(self.defBegin().format(network = str(i), dims = "Dims4{1, mParams.inputC, mParams.inputH, mParams.inputW}"))
                else: 
                    self.add_code(self.defBegin().format(network = str(i), dims = f"mIntermediateDims{i-1}"))
                self.gen_layers(self.backbone[self.LF_bkpt[i]:self.LF_bkpt[i+1]], str(i), 'data')
                if (self.LF_bkpt[i+1] < self.exit_point[i]):
                    self.gen_layers(self.backbone[self.LF_bkpt[i+1]:self.exit_point[i]], str(i), self.backbone[self.LF_bkpt[i]][1] + '->getOutput(0)')
                self.gen_layers(self.exits[i], str(i), self.backbone[self.exit_point[i]-1][1] + '->getOutput(0)')
                self.add_code(self.addOutput(str(i), self.backbone[self.LF_bkpt[i+1]-1][1], dims = f'mIntermediateDims{i}'))
                self.add_code(self.addOutput(str(i), self.exits[i][-1][1]))
                self.add_code(self.defEnd().format(network = str(i), workspace = self.workspace))
            self.add_code(self.defBegin().format(network = str(self.engines-1), dims = f"mIntermediateDims{self.engines-2}"))
            self.gen_layers(self.backbone[self.LF_bkpt[-1]:], str(self.engines-1), 'data')
            self.add_code(self.addOutput(str(self.engines-1), self.backbone[-1][1]))
            self.add_code(self.defEnd().format(network = str(self.engines-1), workspace = self.workspace))

        elif not self.HLayerFusion:
            for i in range (len(self.exits)):
                if i == 0:
                    self.add_code(self.defBegin().format(network = str(i*3), dims = "Dims4{1, mParams.inputC, mParams.inputH, mParams.inputW}"))
                else: 
                    self.add_code(self.defBegin().format(network = str(i*3), dims = f"mIntermediateDims{i*2-1}"))
                self.gen_layers(self.backbone[self.BB_bkpt[i]:self.exit_point[i]], str(i*3), 'data')
                self.add_code(self.addOutput(str(i*3), self.backbone[self.exit_point[i]-1][1], dims = f'mIntermediateDims{i*2}'))
                self.add_code(self.defEnd().format(network = str(i*3), workspace = self.workspace))

                self.add_code(self.defBegin().format(network = str(i*3+1), dims = f"mIntermediateDims{i*2}"))
                self.gen_layers(self.exits[i], str(i*3+1), 'data')
                self.add_code(self.addOutput(str(i*3+1), self.exits[i][-1][1]))
                self.add_code(self.defEnd().format(network = str(i*3+1), workspace = self.workspace))

                self.add_code(self.defBegin().format(network = str(i*3+2), dims = f"mIntermediateDims{i*2}"))
                self.gen_layers(self.backbone[self.exit_point[i]:self.BB_bkpt[i+1]], str(i*3+2), 'data')
                self.add_code(self.addOutput(str(i*3+2), self.backbone[self.BB_bkpt[i+1]-1][1], dims = f'mIntermediateDims{i*2+1}'))
                self.add_code(self.defEnd().format(network = str(i*3+2), workspace = self.workspace))

            self.add_code(self.defBegin().format(network = str(len(self.exits)*3), dims = f"mIntermediateDims{len(self.exits)*2-1}"))
            self.gen_layers(self.backbone[self.BB_bkpt[-1]:], str(len(self.exits)*3), 'data')
            self.add_code(self.addOutput(str(len(self.exits)*3), self.backbone[-1][1]))
            self.add_code(self.defEnd().format(network = str(len(self.exits)*3), workspace = self.workspace))
        else:
            for i in range (len(self.exits)):
                if i == 0:
                    self.add_code(self.defBegin().format(network = str(i*3), dims = "Dims4{1, mParams.inputC, mParams.inputH, mParams.inputW}"))
                else: 
                    self.add_code(self.defBegin().format(network = str(i*3), dims = f"mIntermediateDims{i*3-1}"))
                self.gen_layers(self.backbone[self.BB_bkpt[i]:self.LF_bkpt[i+1]], str(i*3), 'data')
                self.gen_layers(self.exits[i][:(self.LF_bkpt[i+1]-self.exit_point[i])], str(i*3), self.backbone[self.exit_point[i]-1][1] + '->getOutput(0)')
                self.add_code(self.addOutput(str(i*3), self.backbone[self.LF_bkpt[i+1]-1][1], dims = f'mIntermediateDims{i*3}'))
                self.add_code(self.addOutput(str(i*3), self.exits[i][self.LF_bkpt[i+1]-self.exit_point[i]-1][1], dims = f'mIntermediateDims{i*3+1}'))
                self.add_code(self.defEnd().format(network = str(i*3), workspace = self.workspace))

                self.add_code(self.defBegin().format(network = str(i*3+1), dims = f"mIntermediateDims{i*3+1}"))
                self.gen_layers(self.exits[i][self.LF_bkpt[i+1]-self.exit_point[i]:], str(i*3+1), 'data')
                self.add_code(self.addOutput(str(i*3+1), self.exits[i][-1][1]))
                self.add_code(self.defEnd().format(network = str(i*3+1), workspace = self.workspace))

                self.add_code(self.defBegin().format(network = str(i*3+2), dims = f"mIntermediateDims{i*3}"))
                self.gen_layers(self.backbone[self.LF_bkpt[i+1]:self.BB_bkpt[i+1]], str(i*3+2), 'data')
                self.add_code(self.addOutput(str(i*3+2), self.backbone[self.BB_bkpt[i+1]-1][1], dims = f'mIntermediateDims{i*3+2}'))
                self.add_code(self.defEnd().format(network = str(i*3+2), workspace = self.workspace))

            self.add_code(self.defBegin().format(network = str(len(self.exits)*3), dims = f"mIntermediateDims{len(self.exits)*3-1}"))
            self.gen_layers(self.backbone[self.BB_bkpt[-1]:], str(len(self.exits)*3), 'data')
            self.add_code(self.addOutput(str(len(self.exits)*3), self.backbone[-1][1]))
            self.add_code(self.defEnd().format(network = str(len(self.exits)*3), workspace = self.workspace))


    def gen_layers(self, net, net_name, in_layer):
        for i, [layer, name] in enumerate(net):
            self.add_code(eval('self.' + layer).format(
                network = net_name,
                input = in_layer if i==0 else net[i-1][1] + '->getOutput(0)',
                output = name
            ))

    def gen_bindings(self):
        self.mallocBindings = ''
        self.freeBindings = ''
        if not self.multistream:
            self.engines = len(self.exits) + 1
            self.binding_sizes = [3] * self.engines
            self.binding_sizes[-1] = 2
            self.interDims = len(self.exits)

            self.mallocBindings += '    CHECK(cudaMalloc(&binding0[0], mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float)));\n'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding0[1], samplesCommon::volume(mIntermediateDims0) * sizeof(float)));\n'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding0[2], mParams.outputSize * sizeof(float)));\n'
            self.freeBindings += '    CHECK(cudaFree(binding0[0]));\n'
            self.freeBindings += '    CHECK(cudaFree(binding0[1]));\n'
            self.freeBindings += '    CHECK(cudaFree(binding0[2]));\n'
            for i in range (1, len(self.exits)):
                self.mallocBindings += f'    binding{i}[0] = binding{i-1}[1];\n'
                self.mallocBindings += f'    CHECK(cudaMalloc(&binding{i}[1], samplesCommon::volume(mIntermediateDims{i}) * sizeof(float)));\n'
                self.mallocBindings += f'    binding{i}[2] = binding0[2];\n'
                self.freeBindings += f'    CHECK(cudaFree(binding{i}[1]));\n'
            self.mallocBindings += f'    binding{self.engines-1}[0] = binding{self.engines-2}[1];\n'
            self.mallocBindings += f'    binding{self.engines-1}[1] = binding0[2];\n'


        elif not self.HLayerFusion:
            self.engines = len(self.exits) * 3 + 1
            self.binding_sizes = [2] * self.engines
            self.interDims = 2 * len(self.exits)

            self.mallocBindings += '    CHECK(cudaMalloc(&binding0[0], mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float)));\n'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding0[1], samplesCommon::volume(mIntermediateDims0) * sizeof(float)));\n'
            self.mallocBindings += '    binding1[0] = binding0[1];\n'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding1[1], mParams.outputSize * sizeof(float)));\n'
            self.mallocBindings += '    binding2[0] = binding0[1];\n'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding2[1], samplesCommon::volume(mIntermediateDims1) * sizeof(float)));\n'
            self.freeBindings += '    CHECK(cudaFree(binding0[0]));\n'
            self.freeBindings += '    CHECK(cudaFree(binding0[1]));\n'
            self.freeBindings += '    CHECK(cudaFree(binding1[1]));\n'
            self.freeBindings += '    CHECK(cudaFree(binding2[1]));\n'
            for i in range (1, len(self.exits)):
                self.mallocBindings += f'    binding{i*3}[0] = binding{i*3-1}[1];\n'
                self.mallocBindings += f'    CHECK(cudaMalloc(&binding{i*3}[1], samplesCommon::volume(mIntermediateDims{i*2}) * sizeof(float)));\n'
                self.mallocBindings += f'    binding{i*3+1}[0] = binding{i*3}[1];\n'
                self.mallocBindings += f'    binding{i*3+1}[1] = binding1[1];\n'
                self.mallocBindings += f'    binding{i*3+2}[0] = binding{i*3}[1];\n'
                self.mallocBindings += f'    CHECK(cudaMalloc(&binding{i*3+2}[1], samplesCommon::volume(mIntermediateDims{i*2+1}) * sizeof(float)));\n'
                self.freeBindings += f'    CHECK(cudaFree(binding{i*3}[1]));\n'
                self.freeBindings += f'    CHECK(cudaFree(binding{i*3+2}[1]));\n'
            self.mallocBindings += f'    binding{self.engines-1}[0] = binding{self.engines-2}[1];\n'
            self.mallocBindings += f'    binding{self.engines-1}[1] = binding1[1];\n'

        else:
            self.engines = len(self.exits) * 3 + 1
            self.binding_sizes = [2] * self.engines
            for i in range(len(self.exits)):
                self.binding_sizes[i*3] = 3
            self.interDims = 3 * len(self.exits)

            self.mallocBindings += '    CHECK(cudaMalloc(&binding0[0], mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float)));'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding0[1], samplesCommon::volume(mIntermediateDims0) * sizeof(float)));'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding0[2], samplesCommon::volume(mIntermediateDims1) * sizeof(float)));'
            self.mallocBindings += '    binding1[0] = binding0[2];'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding1[1], mParams.outputSize * sizeof(float)));'
            self.mallocBindings += '    binding2[0] = binding0[1];'
            self.mallocBindings += '    CHECK(cudaMalloc(&binding2[1], samplesCommon::volume(mIntermediateDims2) * sizeof(float)));'
            self.freeBindings += '    CHECK(cudaFree(binding0[0]));'
            self.freeBindings += '    CHECK(cudaFree(binding0[1]));'
            self.freeBindings += '    CHECK(cudaFree(binding0[2]));'
            self.freeBindings += '    CHECK(cudaFree(binding1[1]));'
            self.freeBindings += '    CHECK(cudaFree(binding2[1]));'
            for i in range (1, len(self.exits)):
                self.mallocBindings += f'    binding{i*3}[0] = binding{i*3-1}[1];'
                self.mallocBindings += f'    CHECK(cudaMalloc(&binding{i*3}[1], samplesCommon::volume(mIntermediateDims{i*3}) * sizeof(float)));'
                self.mallocBindings += f'    CHECK(cudaMalloc(&binding{i*3}[2], samplesCommon::volume(mIntermediateDims{i*3+1}) * sizeof(float)));'
                self.mallocBindings += f'    binding{i*3+1}[0] = binding{i*3}[2];'
                self.mallocBindings += f'    binding{i*3+1}[1] = binding1[1];'
                self.mallocBindings += f'    binding{i*3+2}[0] = binding{i*3}[1];'
                self.mallocBindings += f'    CHECK(cudaMalloc(&binding{i*3+2}[1], samplesCommon::volume(mIntermediateDims{i*3+2}) * sizeof(float)));'
                self.freeBindings += f'    CHECK(cudaFree(binding{i*3}[1]));'
                self.freeBindings += f'    CHECK(cudaFree(binding{i*3}[2]));'
                self.freeBindings += f'    CHECK(cudaFree(binding{i*3+2}[1]));'
            self.mallocBindings += f'    binding{self.engines-1}[0] = binding{self.engines-2}[1];'
            self.mallocBindings += f'    binding{self.engines-1}[1] = binding1[1];'
    
    def gen_trt(self):
        # C++ functions
        def gen_lib_header():
            # import libraries
            std_libs = ['cuda_runtime_api.h', 'unistd.h', 'cstdlib', 'fstream', 'iostream', 'sstream', 'chrono', 'vector', 'numeric']
            if self.multithread:
                std_libs.extend(['thread', 'atomic'])

            for lib in std_libs:
                self.add_code(f'#include <{lib}>')
            self.add_code('')
            
            trt_libs = ['argsParser.h', 'buffers.h', 'common.h', 'logger.h', 'NvInfer.h']
            for lib in trt_libs:
                self.add_code(f'#include "{lib}"')
            self.add_code('')
            
            self.add_code('using samplesCommon::SampleUniquePtr;')
            self.add_code('typedef std::chrono::high_resolution_clock Clock;', 2)
            
            self.add_code(f'const std::string gSampleName = "TensorRT.B_{self.model_name}_{self.dataset}";')
            if self.multithread:
                self.add_code('std::atomic<int> sig (0);')
        
        def gen_class():
            # construct class
            self.add_code(
'''        
struct EENetParams : public samplesCommon::SampleParams {{
    int inputC;                  // The input channel
    int inputH;                  // The input height
    int inputW;                  // The input width
    int outputSize;              // The output size
    std::string weightsFile;     // The filename of the weights file
    std::string testFile;        // The filename of the test set file
    std::string labelFile;       // The filename of the test label file
}};

class EENet {{
    public:
        EENet(const EENetParams& params) : mParams(params) {{}}

        bool build();                         // Build the network engines
        bool prepare();                       // Prepare engines for test
        bool test(int sample_total);          // Runs the TensorRt inference engine for sample
        bool teardown();                      // Clean up

    private:
        EENetParams mParams;                                                 // Parameters for the sample
        std::map<std::string, nvinfer1::Weights> mWeightMap;                 // The weight name
        std::vector<std::unique_ptr<samplesCommon::HostMemory>> weightsMem;  // Host weights memory holder
    
        void loadTestSample(int num);
        int infer({pee});
        void bb_thread();
        bool validateOutput();
        bool check_exit_criteria(float threshold); 
        std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file); 
        
        int currentLabel;
        bool terminate = false; '''.format(pee = 'float threashold' if self.fix_pee else '')
            )
        
            for i in range (self.engines):
                self.add_code('''
        bool buildEngine{i}(
            const SampleUniquePtr<nvinfer1::IBuilder>& builder,
            const SampleUniquePtr<nvinfer1::IRuntime>& runtime,
            cudaStream_t profileStream
        ); 
        SampleUniquePtr<nvinfer1::ICudaEngine> mEngine{i}{{nullptr}};
        SampleUniquePtr<nvinfer1::IExecutionContext>mContext{i}{{nullptr}}; '''.format(i=i))
        
            self.add_code('        void *input, *output;')
            if self.engines == 1:
                self.add_code('        void *binding0[2];')
            else:
                for i, size in enumerate(self.binding_sizes):
                    self.add_code(f'        void *binding{i}[{size}];')

                for i in range (self.interDims):
                    self.add_code(f'        nvinfer1::Dims mIntermediateDims{i};')

                if self.multistream:
                    self.add_code('        cudaStream_t streamA; cudaStream_t streamB;')

            temp = []
            temp2 = []
            for _ in range (1 if self.engines==1 else len(self.exits)+1):
                temp.append('0')
                temp2.append('{}')
            self.add_code('')
            self.add_code(f'        int accuracy[{1 if self.engines==1 else len(self.exits)+1}] = {{{", ".join(temp)}}};')
            self.add_code(f'        float latency[{1 if self.engines==1 else len(self.exits)+1}] = {{{", ".join(temp)}}};')
            self.add_code(f'        int outputs[{1 if self.engines==1 else len(self.exits)+1}] = {{{", ".join(temp)}}};', 2)
            self.add_code(f'        std::vector<std::vector<float>> latencies = {{{", ".join(temp2)}}};')
            
            self.add_code('''
        template <typename T>
        SampleUniquePtr<T> makeUnique(T* t) {
            return SampleUniquePtr<T>{t};
        }
}; ''')
    
        def gen_build_func():
            # construct build function
            self.add_code(
            '''
bool EENet::build() {
    mWeightMap = loadWeights(locateFile(mParams.weightsFile, mParams.dataDirs));

    auto builder = makeUnique(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder) {
        sample::gLogError << "Create inference builder failed." << std::endl;
        return false;
    }

    auto runtime = makeUnique(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!runtime) {
        sample::gLogError << "Runtime object creation failed." << std::endl;
        return false;
    }

    try {
        // CUDA stream used for profiling by the builder.
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream) {
            return false;
        }
        bool result; ''' )
            temp = []
            for i in range (self.engines):
                temp.append(f'buildEngine{i}(builder, runtime, *profileStream)')
            self.add_code(f'        result = ' + '\n              && '.join(temp) + ';', 0)
            self.add_code(
            '''
        return result;
    } catch (std::runtime_error& e) {
        sample::gLogError << e.what() << std::endl;
        return false;
    }
}       ''', 2)
    
        def gen_prepare_func():
            # construct prepare function
            self.add_code('bool EENet::prepare() {')
            for i in range (self.engines):
                self.add_code(f'    mContext{i} = makeUnique(mEngine{i}->createExecutionContext());')
                self.add_code(f'    if (!mContext{i}) {{')
                self.add_code(f'        sample::gLogError << "Context{i} build failed." << std::endl;')
                self.add_code('        return false; \n    }', 2)

            if self.multistream:
                self.add_code('    CHECK(cudaStreamCreate(&streamA));')
                self.add_code('    CHECK(cudaStreamCreate(&streamB));')

            self.add_code('    CHECK(cudaMallocHost(&input, mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float)));')
            self.add_code('    CHECK(cudaMallocHost(&output, mParams.outputSize * sizeof(float)));')

            if self.engines == 1:
                self.add_code('    CHECK(cudaMalloc(&binding0[0], mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float)));')
                self.add_code('    CHECK(cudaMalloc(&binding0[1], mParams.outputSize * sizeof(float)));')
            else:
                self.add_code(self.mallocBindings)

            self.add_code('    return true;')
            self.add_code('}')
        
        def gen_test_func():
            # constuct test function
            self.add_code('''
bool EENet::test(int sample_total) {
    auto start_time = Clock::now();
    auto end_time = Clock::now();
    int EE;
    float exe_time; ''')

            if self.multithread:
                self.add_code('    terminate = false;')
                self.add_code('    std::thread thread(&EENet::bb_thread, this);')
            
            if self.warmup > 0:
                self.add_code('''
    // warm up
    for (int i = 0; i < {warmup}; i++) {{
        loadTestSample(i);
        infer({pee});
    }}   
        '''.format(warmup = self.warmup, pee = 'i' if self.fix_pee else ''))

            self.add_code('''
    std::cout << "\\n\\nPerforming Experiments\\n";
        
    for (int i = 0; i < sample_total; i++) {{
        loadTestSample(i);

        start_time = Clock::now();
        EE = infer({pee});
        end_time = Clock::now();
        
        exe_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count()/1000000.0;
        {show}

        outputs[EE]++;
        latencies[EE].push_back(exe_time);
        accuracy[EE] += validateOutput();
    }}

    for (unsigned int i = 0; i < latencies.size(); i++) {{
        std::sort(latencies[i].begin(), latencies[i].end());
        latency[i] = std::accumulate(latencies[i].begin() + latencies[i].size() * 0, latencies[i].begin() + latencies[i].size() * 0.5, latency[i]);
        latency[i] /= int(latencies[i].size() * 0.5) - int(latencies[i].size() * 0);
    }}
        '''.format(pee = 'i' if self.fix_pee else '', show = 'std::cout << i+1 << "    Class: " << currentLabel << "    " << std::setprecision(5) << std::fixed << "CPU: " << exe_time << " ms   ";' if self.print_individual else ''))

            if self.multithread:
                self.add_code('    terminate = true;')
                self.add_code('    thread.join();')
            
            self.add_code(f'    float totalLatency = 0; totalLatency = std::accumulate(latency, latency+{1 if self.engines==1 else len(self.exits)+1}, totalLatency);')
            self.add_code(f'    int totalAccuracy = 0; totalAccuracy = std::accumulate(accuracy, accuracy+{1 if self.engines==1 else len(self.exits)+1}, totalAccuracy);')
            self.add_code('    std::cout << "///////////////////////////////////////////////////////////\\nSummary:\\n"')
            self.add_code('              << "Total outputs: " << sample_total')
            for i in range (1 if self.engines==1 else len(self.exits)+1):
                self.add_code(f'              << "    Exit{i}: " << outputs[{i}] * 100.0 / sample_total << "%"')
            self.add_code(f'              << "\\nAverage Latency:  " << totalLatency / {1 if self.engines==1 else len(self.exits)+1} << "ms"')
            for i in range (1 if self.engines==1 else len(self.exits)+1):
                self.add_code(f'              << "    Exit{i}: " << latency[{i}] << "ms"')
            self.add_code('              << "\\nAverage Accuracy: " << totalAccuracy * 100.0 / sample_total << "%"')
            for i in range (1 if self.engines==1 else len(self.exits)+1):
                self.add_code(f'              << "    Exit{i}: " << accuracy[{i}] * 100.0 / outputs[{i}] << "%"')
            self.add_code('              << std::endl;', 2)
            
            # write result to file
            self.add_code('    std::ofstream datafile;')
            self.add_code('    datafile.open("data.txt", std::ios_base::app);')
            
            self.add_code(f'    datafile << totalLatency / {1 if self.engines==1 else len(self.exits)+1} << ", ";')
            for i in range (1 if self.engines==1 else len(self.exits)+1):
                self.add_code(f'    datafile << latency[{i}] << ", ";')
            self.add_code('''
    datafile << std::endl;
    datafile.close();
    return true;
}       ''')
        
        def gen_infer_func():
            # construct infer function
            self.add_code(f'int EENet::infer({"float threshold" if self.fix_pee else ""}) {{')
            if self.multistream:
                self.add_code('    CHECK(cudaMemcpyAsync(binding0[0], input, mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float), cudaMemcpyHostToDevice, streamA));', 2)
            else:
                self.add_code('    CHECK(cudaMemcpy(binding0[0], input, mParams.inputC * mParams.inputW * mParams.inputH * sizeof(float), cudaMemcpyHostToDevice));', 2)
            
            if self.engines == 1:
                self.add_code('    if (!mContext0->executeV2(binding0)) { std::cout << "Execute engine0 error" << std::endl; }')
                self.add_code('    CHECK(cudaMemcpy(output, binding0[1], mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost));', 2)
                self.add_code('    return 0;\n}')
            else:
                for i in range (len(self.exits)+1):
                    indent = '    ' * (i+1)
                    if self.multistream:
                        if i < len(self.exits):
                            self.add_code(indent + f'if (!mContext{i*3}->enqueueV2(binding{i*3}, streamA, nullptr)) {{ std::cout << "Execute engine{i*3} error" << std::endl; }}')
                            self.add_code(indent + 'cudaStreamSynchronize(streamA);')
                            if self.multithread:
                                self.add_code(indent + f'sig = {i+1};')
                                self.add_code(indent + f'if (!mContext{i*3+1}->enqueueV2(binding{i*3+1}, streamA, nullptr)) {{ std::cout << "Execute engine{i*3+1} error" << std::endl; }}')
                            else:
                                self.add_code(indent + f'if (!mContext{i*3+1}->enqueueV2(binding{i*3+1}, streamA, nullptr)) {{ std::cout << "Execute engine{i*3+1} error" << std::endl; }}')
                                self.add_code(indent + f'if (!mContext{i*3+2}->enqueueV2(binding{i*3+2}, streamB, nullptr)) {{ std::cout << "Execute engine{i*3+2} error" << std::endl; }}')
                            self.add_code(indent + f'CHECK(cudaMemcpyAsync(output, binding{i*3+1}[1], mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost, streamA));')
                            self.add_code(indent + 'cudaStreamSynchronize(streamA);')

                            if self.fix_pee:
                                self.add_code(indent + f'if (!check_exit_criteria((bool)(((int)threshold+{i})%{len(self.exits)+1}))) {{')
                            else:
                                self.add_code(indent + f'if (!check_exit_criteria({self.threshold[i]})) {{')
                                
                            if self.multithread:
                                self.add_code(indent + '    while(sig) {continue;}')
                            else:
                                self.add_code(indent + '    cudaStreamSynchronize(streamB);')

                        else:
                            self.add_code(indent + f'if (!mContext{self.engines-1}->enqueueV2(binding{self.engines-1}, streamA, nullptr)) {{ std::cout << "Execute engine{self.engines-1} error" << std::endl; }}')
                            self.add_code(indent + f'CHECK(cudaMemcpyAsync(output, binding{self.engines-1}[1], mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost, streamA));')
                            self.add_code(indent + 'cudaStreamSynchronize(streamA);')
                    else:
                        self.add_code(indent + f'if (!mContext{i}->executeV2(binding{i})) {{ std::cout << "Execute engine{i} error" << std::endl; }}')
                        self.add_code(indent + f'CHECK(cudaMemcpy(output, binding{i}[{2 if i != len(self.exits) else 1}], mParams.outputSize * sizeof(float), cudaMemcpyDeviceToHost));', 2)
                        if i != len(self.exits):
                            if self.fix_pee:
                                self.add_code(indent + f'if (!check_exit_criteria((bool)(((int)threshold+{i})%{len(self.exits)+1}))) {{')
                            else:
                                self.add_code(indent + f'if (!check_exit_criteria({self.threshold[i]})) {{')
                for i in range (len(self.exits)+1):
                    indent = '    '
                    if self.multistream:
                        if self.multithread:
                            self.add_code(indent * (len(self.exits)+1-i) + 'while(sig) {continue;}')
                        else:
                            self.add_code(indent * (len(self.exits)+1-i) + 'cudaStreamSynchronize(streamB);')
                    self.add_code(indent * (len(self.exits)+1-i) + f'return {len(self.exits)+1-i-1};')
                    self.add_code(indent * (len(self.exits)+1-i-1 ) + '}')
            
            if self.multithread:
                self.add_code('void EENet::bb_thread() {')
                self.add_code('    while (!terminate) {')
                for i in range (len(self.exits)):
                    self.add_code('''
        {els}if (sig == {sig}) {{
            if (!mContext{engine}->enqueueV2(binding{engine}, streamB, nullptr)) {{ std::cout << "Execute engine{engine} error" << std::endl; }}
            cudaStreamSynchronize(streamB);
            sig = 0;
        }}
                    '''.format(els = '' if i == 0 else 'else ', sig = i+1, engine = i*3+2))
                self.add_code('    }')
                self.add_code('    std::cout << "Thread finished" << std::endl;')
                self.add_code('}')
    
        def gen_validate_func():
        # construct check_exit_criteria function
            self.add_code('''
bool EENet::check_exit_criteria(float threshold) {
    const float* bufRaw = static_cast<const float*>(output);
    std::vector<float> prob(bufRaw, bufRaw + mParams.outputSize * sizeof(float));
    return prob[std::max_element(prob.begin(), prob.end()) - prob.begin()] > threshold;
}
        ''')
        # construct validateOutput function
            self.add_code('''
bool EENet::validateOutput() {{
    const float* bufRaw = static_cast<const float*>(output);
    std::vector<float> prob(bufRaw, bufRaw + mParams.outputSize * sizeof(float));
    int predictedDigit = std::max_element(prob.begin(), prob.end()) - prob.begin();
    {show}

    return currentLabel == predictedDigit;
}}       '''.format(show = 'std::cout << "Prediction: " << predictedDigit << "    Prob: " << prob[predictedDigit] << std::endl;' if self.print_individual else ''))
    
        def gen_load_sample_func():
            # construct loadTestSample function
            labelFile = 'testFile'
            if self.dataset == 'MNIST':
                labelFile = 'labelFile'
                labelOffset = 8
                labelJump = 1
                dataOffset = 16
                dataJump = 784
                tf = '(x / 255.0)'
            elif self.dataset == 'CIFAR10':
                labelOffset = 0
                labelJump = 3073
                dataOffset = 1
                dataJump = 3073
                tf = '(((x / 255.0) - 0.4733630120754242) / 0.2515689432621002)'
            elif self.dataset == 'ImageNet':
                labelOffset = 0
                # labelJump = 150528
                labelJump = 0
                dataOffset = 1
                # dataJump = 150528
                dataJump = 0
                tf = '(x)'
            
            self.add_code('void EENet::loadTestSample(int num) {')
            if self.dataset == 'MNIST':
                self.add_code('''
    std::ifstream labelFile(locateFile(mParams.labelFile, mParams.dataDirs), std::ifstream::binary);
    ASSERT(labelFile.is_open() && "Attempting to read from a file that is not open."); 
            ''')
            self.add_code('''
    std::ifstream testFile(locateFile(mParams.testFile, mParams.dataDirs), std::ifstream::binary);
    ASSERT(testFile.is_open() && "Attempting to read from a file that is not open."); 

    Dims4 inputDims{{1, mParams.inputC, mParams.inputH, mParams.inputW}};
    size_t vol = samplesCommon::volume(inputDims);
    std::vector<uint8_t> label(1);
    
    // load label
    {labelFile}.seekg({labelOffset} + {labelJump} * num);
    {labelFile}.read(reinterpret_cast<char*>(label.data()), 1);
    currentLabel = (int)label[0];

    //load data
    std::vector<uint8_t> fileData(vol);
    testFile.seekg({dataOffset} + {dataJump} * num);
    testFile.read(reinterpret_cast<char*>(fileData.data()), vol);

    // Normalize and copy to the host buffer.
    float* hostDataBuffer = static_cast<float*>(input);
    std::transform(fileData.begin(), fileData.end(), hostDataBuffer,
        [](uint8_t x) {{ return static_cast<float>{tf}; }});
    return;
}}      '''.format(labelFile = labelFile, labelOffset = labelOffset, labelJump = labelJump, dataOffset = dataOffset, dataJump = dataJump, tf = tf))
    
        def gen_teardown_func():
            # construct teardown function
            self.add_code('bool EENet::teardown() {')
            if self.multistream:
                self.add_code('    cudaStreamDestroy(streamA);')
                self.add_code('    cudaStreamDestroy(streamB);')
            self.add_code('    CHECK(cudaFreeHost(input));')
            self.add_code('    CHECK(cudaFreeHost(output));')
            if self.engines == 1:
                self.add_code('    CHECK(cudaFree(binding0[0]));')
                self.add_code('    CHECK(cudaFree(binding0[1]));')
            else:
                self.add_code(self.freeBindings)
            self.add_code('    return true;\n}')
    
        def gen_load_wts_func():
            # construct loadWeights Function
            self.add_code('''
std::map<std::string, nvinfer1::Weights> EENet::loadWeights(const std::string& file) {
    sample::gLogInfo << "Loading weights: " << file << std::endl;

    // Open weights file
    std::ifstream input(file, std::ios::binary);
    ASSERT(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    ASSERT(count > 0 && "Invalid weight map file.");

    std::map<std::string, nvinfer1::Weights> weightMap;
    while (count--) {
        nvinfer1::Weights wt{DataType::kFLOAT, nullptr, 0};
        int type;
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> type >> size;
        wt.type = static_cast<DataType>(type);

        // Load blob
        if (wt.type == DataType::kFLOAT) {
            // Use uint32_t to create host memory to avoid additional conversion.
            auto mem = new samplesCommon::TypedHostMemory<uint32_t, nvinfer1::DataType::kFLOAT>(size);
            weightsMem.emplace_back(mem);
            uint32_t* val = mem->raw();
            for (uint32_t x = 0; x < size; ++x) {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        } else if (wt.type == DataType::kHALF) {
            // HalfMemory's raw type is uint16_t
            auto mem = new samplesCommon::HalfMemory(size);
            weightsMem.emplace_back(mem);
            auto val = mem->raw();
            for (uint32_t x = 0; x < size; ++x) {
                input >> std::hex >> val[x];
            }
            wt.values = val;
        }
        wt.count = size;
        weightMap[name] = wt;
    }
    return weightMap;
}       ''', 2)
    
        def gen_init_func():
            # construct initializeSampleParams function
            if self.dataset == 'MNIST':
                inputDim = [1, 28, 28]
                outputDim = 10
                testFile = 'MNIST/t10k-images-idx3-ubyte'
                labelFile = 'MNIST/t10k-labels-idx1-ubyte'
            elif self.dataset == 'CIFAR10':
                inputDim = [3, 32, 32]
                outputDim = 10
                testFile = 'CIFAR10/cifar-10-batches-bin/test_batch.bin'
                labelFile = testFile
            elif self.dataset == 'ImageNet':
                inputDim = [3, 224, 224]
                outputDim = 1000
                testFile = 'CIFAR10/cifar-10-batches-bin/test_batch.bin'
                labelFile = testFile
            
            self.add_code('EENetParams initializeSampleParams() {')
            self.add_code('    EENetParams params;')
            self.add_code('    params.dataDirs.push_back("data/");')
            self.add_code(f'    params.inputC = {inputDim[0]};')
            self.add_code(f'    params.inputH = {inputDim[1]};')
            self.add_code(f'    params.inputW = {inputDim[2]};')
            self.add_code(f'    params.outputSize = {outputDim};')
            self.add_code(f'    params.weightsFile = "{self.wtsName}";')
            self.add_code(f'    params.testFile = "{testFile}";')
            self.add_code(f'    params.labelFile = "{labelFile}";')
            self.add_code('    return params;')
            self.add_code('}')
        
        def gen_main_func():
            # construct main fuction
            self.add_code('''
int main(int argc, char** argv) {{
    auto Test = sample::gLogger.defineTest(gSampleName, argc, argv);
    sample::gLogger.reportTestStart(Test);
    EENet sample{{initializeSampleParams()}};

    if (!sample.build()) {{
        return sample::gLogger.reportFail(Test);
    }}
    if (!sample.prepare()) {{
        return sample::gLogger.reportFail(Test);
    }}
    if (!sample.test({sampleTotal})) {{
        return sample::gLogger.reportFail(Test);
    }}

    if (!sample.teardown()) {{
        return sample::gLogger.reportFail(Test);
    }}

    return sample::gLogger.reportPass(Test);
}} '''.format(sampleTotal = 10 if self.nsight else 500 if self.dataset == "ImageNet" else 10000))

        def gen_resnet_func():
            # gen resnet block constructor
            self.add_code('''
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
    // std::cout << "len " << len << std::endl;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + ".conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{stride, stride});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + ".bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + ".conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + ".bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + ".shortcut.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{stride, stride});
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + ".shortcut.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *relu2->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *relu2->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}
            ''')

        self.trt_code = ''
        gen_lib_header()
        gen_class()
        gen_build_func()
        gen_prepare_func()
        gen_test_func()
        gen_infer_func()
        gen_validate_func()
        gen_load_sample_func()
        gen_teardown_func()
        gen_load_wts_func()
        gen_init_func()
        gen_main_func()
        gen_resnet_func()

    # Layer Definitions
    def Conv2d(self, in_channels, out_channels, kernel_size, stride, bias = True, padding = (0,0), wts = ''):
        if bias:
            bias_wts = f'mWeightMap[\"{wts}.bias\"]'
        else:
            bias_wts = 'emptywts'
        layer = f"    // Convolution layer with {out_channels} outputs, {kernel_size[0]}x{kernel_size[1]} filter, {stride[0]}x{stride[1]} stride, {padding[0]}x{padding[1]} padding" + "\n" \
                f"    IConvolutionLayer* {{output}} = Network{{network}}->addConvolutionNd( "                                                    + "\n" \
                f"        *{{input}}, "                                                                                                          + "\n" \
                f"        {out_channels}, "                                                                                                      + "\n" \
                f"        Dims{{{{2, {{{{{kernel_size[0]}, {kernel_size[1]}}}}}}}}}, "                                                           + "\n" \
                f"        mWeightMap[\"{wts}.weight\"], "                                                                                        + "\n" \
                f"        {bias_wts} "                                                                                                           + "\n" \
                f"    );"                                                                                                                        + "\n" \
                f"    ASSERT({{output}});"                                                                                                       + "\n" \
                f"    {{output}}->setStrideNd(Dims{{{{2, {{{{{stride[0]}, {stride[1]}}}}}}}}});"                                                 + "\n" \
                f"    {{output}}->setPaddingNd(Dims{{{{2, {{{{{padding[0]}, {padding[1]}}}}}}}}});"                                              + "\n" \
                f"    {{output}}->setName(\"{{output}}\");"                                                                                      + "\n"
        return layer

    def MaxPool2d(self, kernel_size, stride, padding, dilation, ceil_mode, wts):
        layer = f"    // Max pooling layer with kernel of {kernel_size}x{kernel_size} and stride size of {stride}x{stride}"                      + "\n" \
                f"    IPoolingLayer* {{output}} = Network{{network}}->addPoolingNd("                                                             + "\n" \
                f"        *{{input}},"                                                                                                           + "\n" \
                f"        PoolingType::kMAX,"                                                                                                    + "\n" \
                f"        Dims{{{{2, {{{{{kernel_size}, {kernel_size}}}}}}}}}"                                                                   + "\n" \
                f"    );"                                                                                                                        + "\n" \
                f"    ASSERT({{output}});"                                                                                                       + "\n" \
                f"    {{output}}->setStrideNd(Dims{{{{2, {{{{{stride}, {stride}}}}}}}}});"                                                       + "\n" \
                f"    {{output}}->setName(\"{{output}}\");"                                                                                      + "\n"
        if ceil_mode: layer += f"    {{output}}->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);"                                               + "\n"
        return layer

    def AvgPool2d(self, kernel_size, stride, padding, wts):
        layer = f"    // Average pooling layer with kernel of {kernel_size}x{kernel_size} and stride size of {stride}x{stride}"                      + "\n" \
                f"    IPoolingLayer* {{output}} = Network{{network}}->addPoolingNd("                                                             + "\n" \
                f"        *{{input}},"                                                                                                           + "\n" \
                f"        PoolingType::kAVERAGE,"                                                                                                    + "\n" \
                f"        Dims{{{{2, {{{{{kernel_size}, {kernel_size}}}}}}}}}"                                                                   + "\n" \
                f"    );"                                                                                                                        + "\n" \
                f"    ASSERT({{output}});"                                                                                                       + "\n" \
                f"    {{output}}->setStrideNd(Dims{{{{2, {{{{{stride}, {stride}}}}}}}}});"                                                       + "\n" \
                f"    {{output}}->setName(\"{{output}}\");"                                                                                      + "\n"                                               + "\n"
        return layer

    def ReLU(self, inplace, wts):
        layer = f"    // ReLU activation layer"                                                                                                  + "\n" \
                f"    IActivationLayer* {{output}} = Network{{network}}->addActivation("                                                         + "\n" \
                f"        *{{input}},"                                                                                                           + "\n" \
                f"        ActivationType::kRELU"                                                                                                 + "\n" \
                f"    );"                                                                                                                        + "\n" \
                f"    ASSERT({{output}});"                                                                                                       + "\n" \
                f"    {{output}}->setName(\"{{output}}\");"                                                                                      + "\n"
        return layer

    def Flatten(self, features):
        layer = f"    // Flatten layer flattens t0 {features} features"                                                                          + "\n" \
                f"    IShuffleLayer* {{output}} = Network{{network}}->addShuffle( "                                                              + "\n" \
                f"        *{{input}} "                                                                                                           + "\n" \
                f"    ); "                                                                                                                       + "\n" \
                f"    ASSERT({{output}}); "                                                                                                      + "\n" \
                f"    {{output}}->setReshapeDimensions(Dims{{{{2, {{{{1, {features}}}}}}}}});"                                                   + "\n" \
                f"    {{output}}->setName(\"{{output}}\");"                                                                                      + "\n" 
        return layer

    def Linear(self, in_features, out_features, bias, wts):
        if bias: 
            matmul_layer = "{output}_matmul"
        else :
            matmul_layer = "{output}"

        layer = f"    // Linear layer with {in_features} input features and {out_features} output features"                                      + "\n" \
                f"    IConstantLayer* {{output}}_weights = Network{{network}}->addConstant( "                                                    + "\n" \
                f"        Dims{{{{2, {{{{{out_features}, {in_features}}}}}}}}}, "                                                                + "\n" \
                f"        mWeightMap[\"{wts}.weight\"] "                                                                                         + "\n" \
                f"    ); "                                                                                                                       + "\n" \
                f"    ASSERT({{output}}_weights); "                                                                                              + "\n" \
                f"    {{output}}_weights->setName(\"{{output}}_weights\");"                                                                      + "\n" \
                                                                                                                                                        \
                f"    IMatrixMultiplyLayer* {matmul_layer} = Network{{network}}->addMatrixMultiply("                                             + "\n" \
                f"        *{{input}}, "                                                                                                          + "\n" \
                f"        MatrixOperation::kNONE, "                                                                                              + "\n" \
                f"        *{{output}}_weights->getOutput(0), "                                                                                   + "\n" \
                f"        MatrixOperation::kTRANSPOSE"                                                                                           + "\n" \
                f"    ); "                                                                                                                       + "\n" \
                f"    ASSERT({matmul_layer}); "                                                                                                  + "\n" \
                f"    {matmul_layer}->setName(\"{matmul_layer}\");"                                                                              + "\n" 
        
        if bias:
            layer += f"    IConstantLayer* {{output}}_bias = Network{{network}}->addConstant("                                                   + "\n" \
                     f"        Dims{{{{2, {{{{1, {out_features}}}}}}}}}, "                                                                       + "\n" \
                     f"        mWeightMap[\"{wts}.bias\"]"                                                                                       + "\n" \
                     f"    ); "                                                                                                                  + "\n" \
                     f"    ASSERT({{output}}_bias); "                                                                                            + "\n" \
                     f"    {{output}}_bias->setName(\"{{output}}_bias\");"                                                                       + "\n" \
                                                                                                                                                        \
                     f"    IElementWiseLayer* {{output}} = Network{{network}}->addElementWise("                                                  + "\n" \
                     f"        *{matmul_layer}->getOutput(0), "                                                                                  + "\n" \
                     f"        *{{output}}_bias->getOutput(0), "                                                                                 + "\n" \
                     f"        ElementWiseOperation::kSUM"                                                                                       + "\n" \
                     f"    ); "                                                                                                                  + "\n" \
                     f"    ASSERT({{output}}); "                                                                                                 + "\n" \
                     f"    {{output}}->setName(\"{{output}}\");"                                                                                 + "\n"
        return layer

    def LocalResponseNorm(self, window, alpha, beta, k , wts):
        alpha = f'{float(alpha):f}'
        layer = f"    // Local Response Normalization layer with alpha = {alpha}, beta = {beta} and k = {k}"                                     + "\n" \
                f"    ILRNLayer* {{output}} = Network{{network}}->addLRN( "                                                                      + "\n" \
                f"        *{{input}}, "                                                                                                          + "\n" \
                f"        {window}, "                                                                                                            + "\n" \
                f"        {alpha}, "                                                                                                             + "\n" \
                f"        {beta}, "                                                                                                              + "\n" \
                f"        {k} "                                                                                                                  + "\n" \
                f"    ); "                                                                                                                       + "\n" \
                f"    ASSERT({{output}});"                                                                                                       + "\n" \
                f"    {{output}}->setName(\"{{output}}\");"                                                                                      + "\n"
        return layer + "\n"
    
    def BatchNorm2d(self, channel, eps, momentum, affine, track_running_stats, wts):
        eps = f'{float(eps):f}'
        return f'    IScaleLayer* {{output}} = addBatchNorm2d(Network{{network}}, mWeightMap, *{{input}}, "{wts}", {eps});' + "\n"

    def ResBlock(self, info, wts):
        def blockInfo(in_channel, out_channel, kernel_size, stride, padding, bias):
            return f'    IActivationLayer* {{output}} = basicBlock(Network{{network}}, mWeightMap, *{{input}}, {in_channel}, {out_channel}, {stride[0]}, "{wts}");'
        info = info.split('$')[0]
        info = 'blockInfo' + info[15:]
        return eval(info) +  "\n"

    def Softmax(self):
        layer = f"    // Softmax Layer"                                                                                                          + "\n" \
                f"    ISoftMaxLayer* {{output}} = Network{{network}}->addSoftMax( "                                                              + "\n" \
                f"        *{{input}} "                                                                                                           + "\n" \
                f"    ); "                                                                                                                       + "\n" \
                f"    ASSERT({{output}});"                                                                                                       + "\n" \
                f"    {{output}}->setName(\"{{output}}\");"                                                                                      + "\n" \
                f"    {{output}}->setAxes(2);"                                                                                                   + "\n"
        return layer

    def addOutput(self, network, input, dims = None):
        layer = f"    Network{network}->markOutput(*{input}->getOutput(0));"                                                                     + "\n"
        if dims is not None: layer += f"    {dims} = {input}->getOutput(0)->getDimensions();"                                                    + "\n"

        return layer

    def defBegin(self):
        def_begin = """
bool EENet::buildEngine{network}(
    const SampleUniquePtr<nvinfer1::IBuilder>& builder,
    const SampleUniquePtr<nvinfer1::IRuntime>& runtime, 
    cudaStream_t profileStream) {{

    INetworkDefinition* Network{network} = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
    if (!Network{network}) {{
        sample::gLogError << "Create network failed." << std::endl;
        return false;
    }}

    Weights emptywts{{DataType::kFLOAT, nullptr, 0}};

    ITensor* data = Network{network}->addInput(
        "Input",
        DataType::kFLOAT,
        {dims}
    );
    ASSERT(data);

"""
        return def_begin

    def defEnd(self):
        def_end = """
    auto Config{network} = makeUnique(builder->createBuilderConfig());
    if (!Config{network}) {{
        sample::gLogError << "Create builder config failed." << std::endl;
        return false;
    }}
    Config{network}->setMaxWorkspaceSize({workspace});
    Config{network}->setProfileStream(profileStream);

    SampleUniquePtr<nvinfer1::IHostMemory> Plan{network} = makeUnique(
        builder->buildSerializedNetwork(*Network{network}, *Config{network}));
    if (!Plan{network}) {{
        sample::gLogError << "serialized engine{network} build failed." << std::endl;
        return false;
    }}

    mEngine{network} = makeUnique(
        runtime->deserializeCudaEngine(Plan{network}->data(), Plan{network}->size()));
    if (!mEngine{network}) {{
        sample::gLogError << "engine{network} deserialization failed." << std::endl;
        return false;
    }}

    return true;
}}
"""
        return def_end
    
    def selectLayer(self, exit, A, B):
        layer = '''
    IConstantLayer* criteria{exit_idx} = Network0->addConstant(
        Dims{{2, {{1, 1}}}},
        TH
    );
    criteria{exit_idx}->setName("criteria{exit_idx}");

    ITopKLayer* topEE{exit_idx} = Network0->addTopK(
        *{inputA}->getOutput(0),
        TopKOperation::kMAX,
        1,
        2
    );
    topEE{exit_idx}->setName("topEE{exit_idx}");

    IElementWiseLayer* check{exit_idx} = Network0->addElementWise(
        *topEE{exit_idx}->getOutput(0),
        *criteria{exit_idx}->getOutput(0),
        ElementWiseOperation::kGREATER
    );
    check{exit_idx}->setOutputType(0, DataType::kBOOL);
    check{exit_idx}->getOutput(0)->setType(DataType::kBOOL);
    check{exit_idx}->setName("check_criteria{exit_idx}");

    ISelectLayer* select{exit_idx} = Network0->addSelect(
        *check{exit_idx}->getOutput(0),
        *{inputA}->getOutput(0),
        *{inputB}->getOutput(0)
    );
    select{exit_idx}->setName("select{exit_idx}");
        '''.format(exit_idx = exit, inputA = A, inputB = B)
        return layer