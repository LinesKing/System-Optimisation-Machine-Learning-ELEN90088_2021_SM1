	j?t???j?t???!j?t???	OYS֔?8@OYS֔?8@!OYS֔?8@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$j?t???{?G?z??A?x?&1??Y??(\????*	     ?f@2U
Iterator::Model::ParallelMapV2????Mb??!R?Q?A@)????Mb??1R?Q?A@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapT㥛? ??!??N??NA@)T㥛? ??1??N??NA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatX9??v???!?p?p1@)X9??v???1?p?p1@:Preprocessing2F
Iterator::Model{?G?z??!?_??_?E@)????Mb??1R?Q?!@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mb??!R?Q?@)????Mb??1R?Q?@:Preprocessing2T
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 24.7% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t11.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9NYS֔?8@I?)kʚ?R@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{?G?z??{?G?z??!{?G?z??      ??!       "      ??!       *      ??!       2	?x?&1???x?&1??!?x?&1??:      ??!       B      ??!       J	??(\??????(\????!??(\????R      ??!       Z	??(\??????(\????!??(\????b      ??!       JCPU_ONLYYNYS֔?8@b q?)kʚ?R@