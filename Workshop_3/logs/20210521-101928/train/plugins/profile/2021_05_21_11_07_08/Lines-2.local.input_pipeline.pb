	???Mb?????Mb??!???Mb??	v*d	3)@v*d	3)@!v*d	3)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???Mb??T㥛? ??A5^?I??Y?MbX9??*	     ?Y@2U
Iterator::Model::ParallelMapV29??v????!}}}}}}9@)9??v????1}}}}}}9@:Preprocessing2F
Iterator::Model????????!??????H@)?~j?t???1??????7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!}}}}}}9@)?I+???1??????5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;?O??n??!??????1@);?O??n??1??????!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice;?O??n??!??????!@);?O??n??1??????!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip9??v????!}}}}}}I@){?G?zt?1??????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????Mbp?!______@)????Mbp?1______@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/?$???!??????4@)?~j?t?h?1??????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 12.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t20.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9w*d	3)@I??~Ӟ?U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T㥛? ??T㥛? ??!T㥛? ??      ??!       "      ??!       *      ??!       2	5^?I??5^?I??!5^?I??:      ??!       B      ??!       J	?MbX9???MbX9??!?MbX9??R      ??!       Z	?MbX9???MbX9??!?MbX9??b      ??!       JCPU_ONLYYw*d	3)@b q??~Ӟ?U@