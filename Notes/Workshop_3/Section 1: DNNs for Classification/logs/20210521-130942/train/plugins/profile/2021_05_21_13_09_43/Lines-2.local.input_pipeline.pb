	?MbX9???MbX9??!?MbX9??	??	??Z2@??	??Z2@!??	??Z2@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?MbX9???MbX9??A+??????YV-???*	     ?h@2U
Iterator::Model::ParallelMapV2??ʡE???!I?$I??C@)??ʡE???1I?$I??C@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?Q?????!?m۶m?A@)?Q?????1?m۶m?A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???S㥛?!b}h??+@)9??v????1??>4և*@:Preprocessing2F
Iterator::ModelZd;?O???!^N??)xG@)???Q???1?????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!$I?$I?@)y?&1?|?1$I?$I?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!????S??)????MbP?1????S??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 18.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t12.5 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??	??Z2@Iґ=QiT@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?MbX9???MbX9??!?MbX9??      ??!       "      ??!       *      ??!       2	+??????+??????!+??????:      ??!       B      ??!       J	V-???V-???!V-???R      ??!       Z	V-???V-???!V-???b      ??!       JCPU_ONLYY??	??Z2@b qґ=QiT@