	\???(\??\???(\??!\???(\??	ˠT?x'@ˠT?x'@!ˠT?x'@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$\???(\??9??v????A??Q????Y?V-??*	     ?]@2F
Iterator::Model?~j?t???!?Iݗ?VD@)?~j?t???1?Iݗ?VD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!?؊???B@)+??????1??c+??@@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap;?O??n??!???c+?.@);?O??n??1???c+?.@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mb??!'u_@)????Mb??1'u_@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!"h8???@){?G?zt?1"h8???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 11.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9ˠT?x'@I??k??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	9??v????9??v????!9??v????      ??!       "      ??!       *      ??!       2	??Q??????Q????!??Q????:      ??!       B      ??!       J	?V-???V-??!?V-??R      ??!       Z	?V-???V-??!?V-??b      ??!       JCPU_ONLYYˠT?x'@b q??k??V@