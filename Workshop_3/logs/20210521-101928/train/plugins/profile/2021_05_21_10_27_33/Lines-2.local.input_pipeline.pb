	m???????m???????!m???????	!NnP5<!@!NnP5<!@!!NnP5<!@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$m????????E???Ը?A??|?5^??Y?~j?t???*	     @T@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatL7?A`???!?%???^D@)y?&1???1?????HA@:Preprocessing2U
Iterator::Model::ParallelMapV2??~j?t??!?a?2?t7@)??~j?t??1?a?2?t7@:Preprocessing2F
Iterator::Model???S㥛?!??????@@)????Mb??1???E??#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatey?&1???!?????H1@)????Mb??1???E??#@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!?Kh/?@)?~j?t?x?1?Kh/?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{?G?zt?!e?????@){?G?zt?1e?????@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???S㥫?!??????P@)????Mbp?1???E??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?? ?rh??!"e????4@)?~j?t?h?1?Kh/?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t17.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9!NnP5<!@I<6?Uy?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?E???Ը??E???Ը?!?E???Ը?      ??!       "      ??!       *      ??!       2	??|?5^????|?5^??!??|?5^??:      ??!       B      ??!       J	?~j?t????~j?t???!?~j?t???R      ??!       Z	?~j?t????~j?t???!?~j?t???b      ??!       JCPU_ONLYY!NnP5<!@b q<6?Uy?V@