	????S??????S??!????S??	?\??o?;@?\??o?;@!?\??o?;@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????S??Zd;?O???A??C?l???Y?|?5^???*	      v@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatesh??|???!u?E]?G@)'1?Z??1F]tѕF@:Preprocessing2U
Iterator::Model::ParallelMapV2L7?A`???!     ?B@)L7?A`???1     ?B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Q?????!??.???#@)?? ?rh??1?E]tQ#@:Preprocessing2F
Iterator::Model?O??n??!?袋.*E@)?? ?rh??1?E]tQ@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?|?!?E]t???)y?&1?|?1?E]t???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap/?$???!]t?E?G@)????Mb`?1/?袋.??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor????MbP?!/?袋.??)????MbP?1/?袋.??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 27.6% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*moderate2t12.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?\??o?;@I??dR@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Zd;?O???Zd;?O???!Zd;?O???      ??!       "      ??!       *      ??!       2	??C?l?????C?l???!??C?l???:      ??!       B      ??!       J	?|?5^????|?5^???!?|?5^???R      ??!       Z	?|?5^????|?5^???!?|?5^???b      ??!       JCPU_ONLYY?\??o?;@b q??dR@