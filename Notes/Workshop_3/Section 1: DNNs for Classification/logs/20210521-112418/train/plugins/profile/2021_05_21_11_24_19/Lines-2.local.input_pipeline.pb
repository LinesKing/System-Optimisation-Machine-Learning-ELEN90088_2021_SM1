	?E???????E??????!?E??????	??,O"S7@??,O"S7@!??,O"S7@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?E???????????K??AT㥛? ??Y^?I+??*	     @f@2U
Iterator::Model::ParallelMapV2??ʡE???!?P?"?E@)??ʡE???1?P?"?E@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Q???!NmjS??@@)???Q???1NmjS??@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat;?O??n??!??0?9$@)?? ?rh??1??g<?#@:Preprocessing2F
Iterator::ModelZd;?O???!??^???I@)???Q???1NmjS?? @:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mb??! ??G??@)????Mb??1 ??G??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?! ??G????)????MbP?1 ??G????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 23.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t11.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??,O"S7@I??4l7+S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????K???????K??!?????K??      ??!       "      ??!       *      ??!       2	T㥛? ??T㥛? ??!T㥛? ??:      ??!       B      ??!       J	^?I+??^?I+??!^?I+??R      ??!       Z	^?I+??^?I+??!^?I+??b      ??!       JCPU_ONLYY??,O"S7@b q??4l7+S@