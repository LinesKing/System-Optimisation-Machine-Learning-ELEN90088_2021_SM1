	?K7?A`???K7?A`??!?K7?A`??	?W?s??#@?W?s??#@!?W?s??#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?K7?A`??{?G?z??A?V-??Y???S㥫?*	     ?R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?~j?t???!v?)?Y7@@)Zd;?O???1L?Ϻ??@:Preprocessing2U
Iterator::Model::ParallelMapV2{?G?z??!o0E>?;@){?G?z??1o0E>?;@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???Q???!S??n0E4@)???Q???1S??n0E4@:Preprocessing2F
Iterator::ModelV-???!?)?Y7?C@);?O??n??11E>?S(@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?~j?t?x?!v?)?Y7 @)?~j?t?x?1v?)?Y7 @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!?Y7?"???)????MbP?1?Y7?"???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.9% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t14.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?W?s??#@I
??1??V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	{?G?z??{?G?z??!{?G?z??      ??!       "      ??!       *      ??!       2	?V-???V-??!?V-??:      ??!       B      ??!       J	???S㥫????S㥫?!???S㥫?R      ??!       Z	???S㥫????S㥫?!???S㥫?b      ??!       JCPU_ONLYY?W?s??#@b q
??1??V@