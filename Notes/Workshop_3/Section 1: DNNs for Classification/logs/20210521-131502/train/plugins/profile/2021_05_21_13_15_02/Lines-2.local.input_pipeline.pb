	!?rh????!?rh????!!?rh????	?Z?B}4@?Z?B}4@!?Z?B}4@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$!?rh??????~j?t??A?"??~j??Y???x?&??*	     ?f@2U
Iterator::Model::ParallelMapV2{?G?z??!'<?ߠ?E@){?G?z??1'<?ߠ?E@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap????Mb??!Sc??|A@)????Mb??1Sc??|A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??~j?t??!?E????$@);?O??n??1?Oq??#@:Preprocessing2F
Iterator::Model+??η?!T\2?hI@)9??v????1f?"Qj@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!??Z9??@)y?&1?|?1??Z9??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!Sc??|??)????MbP?1Sc??|??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 20.5% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t11.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?Z?B}4@I~)]??S@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??~j?t????~j?t??!??~j?t??      ??!       "      ??!       *      ??!       2	?"??~j???"??~j??!?"??~j??:      ??!       B      ??!       J	???x?&?????x?&??!???x?&??R      ??!       Z	???x?&?????x?&??!???x?&??b      ??!       JCPU_ONLYY?Z?B}4@b q~)]??S@