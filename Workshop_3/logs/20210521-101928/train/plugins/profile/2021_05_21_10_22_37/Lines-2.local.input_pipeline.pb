	B`??"???B`??"???!B`??"???	z???%%@z???%%@!z???%%@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$B`??"????~j?t???A?n?????Yh??|?5??*	     @U@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9??v????!??????>@)9??v????1??????>@:Preprocessing2F
Iterator::Model{?G?z??!??????G@)?I+???1??????9@:Preprocessing2U
Iterator::Model::ParallelMapV2;?O??n??!------5@);?O??n??1------5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy?&1???!xxxxxx0@)y?&1???1xxxxxx0@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice{?G?zt?!??????@){?G?zt?1??????@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9z???%%@I1m_(L[V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?~j?t????~j?t???!?~j?t???      ??!       "      ??!       *      ??!       2	?n??????n?????!?n?????:      ??!       B      ??!       J	h??|?5??h??|?5??!h??|?5??R      ??!       Z	h??|?5??h??|?5??!h??|?5??b      ??!       JCPU_ONLYYz???%%@b q1m_(L[V@